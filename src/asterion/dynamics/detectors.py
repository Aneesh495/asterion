from dataclasses import asdict, dataclass

import numpy as np

from asterion.contracts.errors import ContractError


@dataclass(frozen=True)
class ValidityPolicy:
    minimum_distance: float = 0.35
    maximum_force: float = 100.0
    maximum_uncertainty: float | None = None
    maximum_bond_ratio: float = 2.0
    bond_reference_radius: float = 1.8
    supported_elements: tuple[int, ...] = (1, 6, 7, 8)

    def validate(self):
        if (
            min(
                self.minimum_distance,
                self.maximum_force,
                self.maximum_bond_ratio,
                self.bond_reference_radius,
            )
            <= 0
        ):
            raise ContractError("validity limits must be positive")
        if self.maximum_uncertainty is not None and self.maximum_uncertainty <= 0:
            raise ContractError("uncertainty threshold must be positive")
        return self


class Detector:
    def __init__(self, numbers, initial_positions, policy: ValidityPolicy):
        self.policy = policy.validate()
        self.numbers = np.asarray(numbers)
        self.bonds = []
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                distance = np.linalg.norm(initial_positions[i] - initial_positions[j])
                if distance < policy.bond_reference_radius:
                    self.bonds.append((i, j, float(distance)))

    def geometry(self, positions):
        if positions.shape != (len(self.numbers), 3) or not np.isfinite(positions).all():
            return "nonfinite-or-invalid-geometry"
        if not set(self.numbers.tolist()) <= set(self.policy.supported_elements):
            return "unsupported-chemistry"
        for i in range(len(positions)):
            distances = np.linalg.norm(positions[i + 1 :] - positions[i], axis=-1)
            if np.any(distances < self.policy.minimum_distance):
                return "atomic-collision"
        for i, j, reference in self.bonds:
            if (
                np.linalg.norm(positions[i] - positions[j])
                > reference * self.policy.maximum_bond_ratio
            ):
                return "bond-domain-exceeded"
        return None

    def forces(self, result):
        if not np.isfinite(result.energy) or not np.isfinite(result.forces).all():
            return "nonfinite-energy-or-force"
        if result.forces.shape != (len(self.numbers), 3):
            return "invalid-force-shape"
        if np.linalg.norm(result.forces, axis=-1).max() > self.policy.maximum_force:
            return "extreme-force"
        if not np.isfinite(result.uncertainty) or result.uncertainty < 0:
            return "invalid-uncertainty"
        if (
            self.policy.maximum_uncertainty is not None
            and result.uncertainty > self.policy.maximum_uncertainty
        ):
            return "uncertainty-domain-exceeded"
        return None

    def to_dict(self):
        return {"policy": asdict(self.policy), "initial_bonds": self.bonds}
