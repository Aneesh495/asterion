from dataclasses import dataclass

import numpy as np
import torch

from asterion.contracts.errors import ContractError, IntegrityError
from asterion.training.identity import state_fingerprint
from asterion.uncertainty.moments import conformal_parameters, member_statistics


@dataclass
class Disagreement:
    energy_mean: np.ndarray
    forces_mean: np.ndarray
    energy_std: np.ndarray
    force_component_std: np.ndarray
    force_score: np.ndarray
    identities: np.ndarray
    member_forces: np.ndarray
    member_energies: np.ndarray


class Committee:
    def __init__(self, models, identities, seeds, forbidden_rows=()):
        if (
            len(models) != 3
            or len(seeds) != 3
            or len(identities) != 3
            or len(set(seeds)) != 3
            or len(set(identities)) != 3
        ):
            raise ContractError("a committee requires three independently seeded trained models")
        if any(model.config.kind == "force_head" for model in models):
            raise ContractError("nonconservative models cannot supply the production committee")
        if len({id(model) for model in models}) != 3:
            raise IntegrityError("committee members alias the same model")
        self.states = []
        self.forbidden_identities: set[str] = set()
        permitted = None
        task = (
            models[0].config.level,
            tuple(models[0].config.elements),
            models[0].config.composition,
        )
        for model, identity, seed in zip(models, identities, seeds, strict=True):
            parameter = next(model.parameters())
            if parameter.dtype not in {torch.float32, torch.float64}:
                raise ContractError("committee requires audited inference precision")
            if (model.config.level, tuple(model.config.elements), model.config.composition) != task:
                raise IntegrityError("committee members disagree on their physical task")
            provenance = getattr(model, "_asterion_provenance", None)
            if (
                provenance is None
                or provenance["checkpoint"].get("origin") != "original-force-loss-training"
                or provenance["checkpoint"].get("updates", 0) <= 0
            ):
                raise IntegrityError("committee requires saved independently trained checkpoints")
            if (
                provenance["checkpoint"]["identity"] != identity
                or provenance["checkpoint"]["seed"] != seed
            ):
                raise IntegrityError("committee seed or identity differs from saved training")
            current = state_fingerprint(model)
            if current != provenance["execution_state"]:
                raise IntegrityError("committee weights changed after checkpoint loading")
            self.states.append(current)
            boundary = provenance["boundary"]
            if boundary is None:
                raise IntegrityError("committee has no retained label-role boundary")
            self.forbidden_identities.update((*boundary["fit"], *boundary["validation"]))
            rows = set(boundary["calibration"])
            permitted = rows if permitted is None else permitted & rows
        if len(set(self.states)) != 3:
            raise IntegrityError("committee member weights are identical")
        if permitted is None:
            raise IntegrityError("committee has no shared calibration boundary")
        self.calibration_identities = frozenset(permitted)
        self.models = tuple(models)
        self.identities = tuple(identities)
        self.seeds = tuple(seeds)
        self.forbidden_rows = set(forbidden_rows)

    def infer(self, geometries, batch_size=8):
        # Labels are unnecessary here. Synthetic zero labels would create false provenance,
        # so coordinates use a separate direct inference loop.
        from asterion.contracts.geometry import collate

        if type(batch_size) is not int or batch_size < 1 or not geometries:
            raise ContractError(
                "committee requires a nonempty geometry list and positive batch size"
            )
        if len(geometries) > 10000 or sum(len(g.numbers) for g in geometries) > 1048576:
            raise ContractError("committee output allocation budget exceeded")
        if len({g.identity for g in geometries}) != len(geometries):
            raise ContractError("committee inference repeats geometry identities")
        if any(
            state_fingerprint(model) != state
            for model, state in zip(self.models, self.states, strict=True)
        ):
            raise IntegrityError("committee weights changed during its lifetime")

        energies, forces = [], []
        for model in self.models:
            parameter = next(model.parameters())
            member_energy, member_force = [], []
            for start in range(0, len(geometries), batch_size):
                batch = collate(
                    geometries[start : start + batch_size],
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                e, f = model(batch)
                member_energy.append(e.detach().cpu().numpy())
                member_force.append(f.detach().cpu().numpy())
            if not member_energy:
                raise ContractError("committee cannot evaluate empty geometries")
            energies.append(np.concatenate(member_energy))
            forces.append(np.concatenate(member_force))
        energy_stack, force_stack = np.stack(energies), np.stack(forces)
        if not np.isfinite(energy_stack).all() or not np.isfinite(force_stack).all():
            raise ContractError("nonfinite committee member output")
        if any(
            state_fingerprint(model) != state
            for model, state in zip(self.models, self.states, strict=True)
        ):
            raise IntegrityError("committee weights changed during inference")
        energy_mean, force_mean, energy_std, force_std, scores = member_statistics(
            energy_stack, force_stack, [len(g.numbers) for g in geometries]
        )
        return Disagreement(
            energy_mean,
            force_mean,
            energy_std,
            force_std,
            scores,
            np.array([g.identity for g in geometries]),
            force_stack,
            energy_stack,
        )

    def calibrate(self, labels, alpha=0.1):
        identities = {label.geometry.identity for label in labels}
        if identities & self.forbidden_identities or not identities <= self.calibration_identities:
            raise IntegrityError("calibration is outside retained held-out label roles")
        rows = {int(label.geometry.identity.rsplit(":", 1)[1]) for label in labels}
        if rows & self.forbidden_rows:
            raise IntegrityError("calibration overlaps committee training or selection rows")
        output = self.infer([label.geometry for label in labels])
        pointers = np.r_[0, np.cumsum([len(label.geometry.numbers) for label in labels])]
        targets = np.concatenate([label.forces for label in labels])
        errors = np.array(
            [
                np.abs(output.forces_mean[a:b] - targets[a:b]).mean()
                for a, b in zip(pointers[:-1], pointers[1:])
            ]
        )
        return (
            Calibration.fit(
                output.force_score, errors, alpha, self.identities, output.identities.tolist()
            ),
            output,
            errors,
        )


@dataclass(frozen=True)
class Calibration:
    multiplier: float
    floor: float
    alpha: float
    sample_count: int
    committee: tuple[str, ...]
    identities: tuple[str, ...]

    def __post_init__(self):
        if (
            not np.isfinite(self.multiplier)
            or self.multiplier < 0
            or not np.isfinite(self.floor)
            or self.floor <= 0
            or not 0 < self.alpha < 1
        ):
            raise ContractError("invalid finite calibration parameters")
        if (
            type(self.sample_count) is not int
            or self.sample_count < 10
            or len(self.identities) != self.sample_count
            or len(set(self.identities)) != self.sample_count
        ):
            raise ContractError("calibration requires unique held-out identities")
        if len(self.committee) != 3 or len(set(self.committee)) != 3:
            raise ContractError("calibration requires three distinct member identities")
        if not all(
            isinstance(identity, str) and identity
            for identity in (*self.committee, *self.identities)
        ):
            raise ContractError("calibration identities must be nonempty strings")

    @staticmethod
    def fit(scores, errors, alpha, committee, identities):
        scores, errors = np.asarray(scores), np.asarray(errors)
        if (
            scores.ndim != 1
            or not 0 < alpha < 1
            or len(scores) < 10
            or scores.shape != errors.shape
        ):
            raise ContractError("calibration requires >=10 geometries and a valid coverage target")
        if (
            not np.isfinite(scores).all()
            or not np.isfinite(errors).all()
            or np.any(scores < 0)
            or np.any(errors < 0)
        ):
            raise ContractError("invalid calibration values")
        multiplier, floor = conformal_parameters(scores, errors, alpha)
        return Calibration(
            multiplier, floor, alpha, len(scores), tuple(committee), tuple(identities)
        )

    def bounds(self, scores):
        scores = np.asarray(scores)
        if scores.ndim != 1 or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ContractError("finite nonnegative force-disagreement vector required")
        try:
            with np.errstate(over="raise", invalid="raise"):
                bounds = self.multiplier * np.maximum(scores, self.floor)
        except FloatingPointError as error:
            raise ContractError("calibrated bound exceeds finite physical output range") from error
        if not np.isfinite(bounds).all():
            raise ContractError("nonfinite calibrated force-error bound")
        return bounds
