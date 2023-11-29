from dataclasses import dataclass

import numpy as np

from asterion.contracts.errors import ContractError


@dataclass
class ForceResult:
    energy: float
    forces: np.ndarray
    uncertainty: float = 0.0
    neighbor_rebuilds: int = 0
    uncertainty_available: bool = False
    uncertainty_quantity: str = "unavailable"


UNCERTAINTY_CODES = {"unavailable": 0, "force-disagreement": 1, "calibrated-mae-bound": 2}


class Harmonic:
    def __init__(self, equilibrium: np.ndarray, stiffness=1.0):
        self.equilibrium = np.asarray(equilibrium, dtype=np.float64)
        self.stiffness = stiffness
        if stiffness <= 0:
            raise ContractError("harmonic stiffness must be positive")

    def __call__(self, positions, numbers):
        displacement = positions - self.equilibrium
        return ForceResult(
            float(0.5 * self.stiffness * np.sum(displacement**2)), -self.stiffness * displacement
        )


class HarmonicBond:
    def __init__(self, distance=1.0, stiffness=1.0):
        self.distance = distance
        self.stiffness = stiffness

    def __call__(self, positions, numbers):
        if len(positions) != 2:
            raise ContractError("harmonic bond requires two atoms")
        delta = positions[1] - positions[0]
        radius = np.linalg.norm(delta)
        if radius < 1e-8:
            raise ContractError("coincident harmonic atoms")
        stretch = radius - self.distance
        first = self.stiffness * stretch * delta / radius
        return ForceResult(float(0.5 * self.stiffness * stretch**2), np.array([first, -first]))


class LennardJones:
    def __init__(self, epsilon=0.01, sigma=1.0):
        if epsilon <= 0 or sigma <= 0:
            raise ContractError("Lennard-Jones parameters must be positive")
        self.epsilon, self.sigma = epsilon, sigma

    def __call__(self, positions, numbers):
        force = np.zeros_like(positions)
        energy = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                delta = positions[j] - positions[i]
                radius = np.linalg.norm(delta)
                if radius < 1e-8:
                    raise ContractError("coincident Lennard-Jones atoms")
                sixth = (self.sigma / radius) ** 6
                energy += 4 * self.epsilon * (sixth**2 - sixth)
                first = 24 * self.epsilon * (sixth - 2 * sixth**2) * delta / radius**2
                force[i] += first
                force[j] -= first
        return ForceResult(float(energy), force)
