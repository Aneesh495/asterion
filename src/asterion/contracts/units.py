"""Physical units derived from SI definitions, never fitted to trajectory behavior."""

import numpy as np

from asterion.contracts.errors import ContractError

ELECTRON_VOLT_J = 1.602176634e-19
AVOGADRO = 6.02214076e23
ATOMIC_MASS_KG = 1.66053906892e-27
ANGSTROM_M = 1e-10
FEMTOSECOND_S = 1e-15
BOLTZMANN_J_K = 1.380649e-23
KCAL_MOL_TO_EV = 4184.0 / AVOGADRO / ELECTRON_VOLT_J
ACCELERATION = ELECTRON_VOLT_J * FEMTOSECOND_S**2 / (ATOMIC_MASS_KG * ANGSTROM_M**2)
KINETIC_ENERGY = 1.0 / ACCELERATION
BOLTZMANN_EV_K = BOLTZMANN_J_K / ELECTRON_VOLT_J
MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O"}


def masses(numbers: np.ndarray) -> np.ndarray:
    try:
        return np.array([MASSES[int(z)] for z in numbers], dtype=np.float64)
    except KeyError as error:
        raise ContractError(f"unsupported mass for element {error.args[0]}") from error


def convert_labels(energy: np.ndarray, force: np.ndarray, units: str):
    if units == "kcal/mol":
        return energy * KCAL_MOL_TO_EV, force * KCAL_MOL_TO_EV
    if units == "eV":
        return energy.copy(), force.copy()
    raise ContractError(f"unknown energy/force units {units}")


def kinetic_energy(velocity: np.ndarray, mass: np.ndarray) -> float:
    if velocity.shape != (len(mass), 3):
        raise ContractError("velocity must have shape [atoms,3]")
    return float(0.5 * KINETIC_ENERGY * np.sum(mass[:, None] * velocity**2))


def temperature(velocity: np.ndarray, mass: np.ndarray, removed_dof: int = 3) -> float:
    dof = 3 * len(mass) - removed_dof
    if dof <= 0:
        raise ContractError("temperature requires positive degrees of freedom")
    return 2 * kinetic_energy(velocity, mass) / (dof * BOLTZMANN_EV_K)


def thermal_velocities(numbers: np.ndarray, kelvin: float, seed: int) -> np.ndarray:
    if not np.isfinite(kelvin) or kelvin < 0 or len(numbers) < 2:
        raise ContractError("thermal initialization requires >=2 atoms and nonnegative temperature")
    mass = masses(numbers)
    rng = np.random.default_rng(seed)
    scale = np.sqrt(BOLTZMANN_EV_K * kelvin * ACCELERATION / mass)
    velocity = rng.normal(size=(len(numbers), 3)) * scale[:, None]
    velocity -= np.sum(mass[:, None] * velocity, axis=0) / mass.sum()
    observed = temperature(velocity, mass)
    if kelvin > 0 and observed > 0:
        velocity *= np.sqrt(kelvin / observed)
    return velocity
