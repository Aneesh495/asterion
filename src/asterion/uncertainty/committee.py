from dataclasses import dataclass

import numpy as np

from asterion.contracts.errors import ContractError, IntegrityError


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
        if len(models) != 3 or len(set(seeds)) != 3 or len(set(identities)) != 3:
            raise ContractError("a committee requires three independently seeded trained models")
        if any(model.config.kind == "force_head" for model in models):
            raise ContractError("nonconservative models cannot supply the production committee")
        self.models = models
        self.identities = tuple(identities)
        self.seeds = tuple(seeds)
        self.forbidden_rows = set(forbidden_rows)

    def infer(self, geometries, batch_size=8):
        # Labels are unnecessary here. Synthetic zero labels would create false provenance,
        # so coordinates use a separate direct inference loop.
        from asterion.contracts.geometry import collate

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
        force_std = force_stack.std(axis=0, ddof=1)
        pointers = np.r_[0, np.cumsum([len(g.numbers) for g in geometries])]
        scores = np.array(
            [np.sqrt(np.mean(force_std[a:b] ** 2)) for a, b in zip(pointers[:-1], pointers[1:])]
        )
        return Disagreement(
            energy_stack.mean(0),
            force_stack.mean(0),
            energy_stack.std(0, ddof=1),
            force_std,
            scores,
            np.array([g.identity for g in geometries]),
            force_stack,
            energy_stack,
        )

    def calibrate(self, labels, alpha=0.1):
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

    @staticmethod
    def fit(scores, errors, alpha, committee, identities):
        scores, errors = np.asarray(scores), np.asarray(errors)
        if not 0 < alpha < 1 or len(scores) < 10 or scores.shape != errors.shape:
            raise ContractError("calibration requires >=10 geometries and a valid coverage target")
        if not np.isfinite(scores).all() or not np.isfinite(errors).all() or np.any(scores < 0):
            raise ContractError("invalid calibration values")
        floor = max(float(np.median(scores) * 0.01), 1e-8)
        ratios = errors / np.maximum(scores, floor)
        rank = min(len(ratios), int(np.ceil((len(ratios) + 1) * (1 - alpha))))
        multiplier = float(np.sort(ratios)[rank - 1])
        return Calibration(
            multiplier, floor, alpha, len(scores), tuple(committee), tuple(identities)
        )

    def bounds(self, scores):
        return self.multiplier * np.maximum(scores, self.floor)
