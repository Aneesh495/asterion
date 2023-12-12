# Units and dimensional mechanics

Internal energy is eV, length angstrom, mass atomic mass units and time femtoseconds. Forces have
units eV/angstrom. rMD17 and MD22 energy and force labels are converted from kcal/mol using
4184 J/kcal divided by Avogadro's constant and the electron-volt definition. Coordinates need no
scale change. Original labels remain in ignored author files.

One force unit acting on one mass unit gives acceleration

`a = (1.602176634e-19 / 1.66053906892e-27) * 1e-10 * F / m`

in angstrom/fs squared. The conversion coefficient is 0.009648533202185. The reciprocal converts
mass times squared angstrom/fs velocity into eV. Boltzmann's constant is obtained from its SI
definition divided by the electron-volt definition. Initial velocities have their center of mass
motion removed and are rescaled to the declared kinetic temperature with 3N-3 degrees of freedom.
Rotational degrees of freedom remain included in this initialization contract.

The reference checker repeats these dimensional calculations in SI instead of importing the
production acceleration or kinetic-energy helpers. The current masses are ordinary isotopic
averages, not isotope-resolved masses. Periodic cells, nonzero charges and nonsinglet multiplicities
are rejected at the geometry boundary.

Internal kinetic temperature removes the mass-weighted center-of-mass velocity before evaluating
kinetic energy and uses 3N minus 3 degrees of freedom. A Galilean boost therefore leaves internal
temperature unchanged. `temperature(..., removed_dof=0)` instead reports total kinetic temperature
using all 3N Cartesian degrees. Other constraint counts are unsupported. The single-atom internal
temperature is undefined; trajectory snapshots retain its zero placeholder rather than claiming
an internal thermal degree. Thermal initialization removes center-of-mass momentum and rescales
once to its declared initial temperature. This fixed-kinetic-energy initialization is distinct
from an unconstrained Maxwell sample. The BAOAB thermostat still acts on all atomic velocities;
its recorded heat includes center-of-mass kinetic changes. Independent replay computes internal
temperature from SI masses, velocities and Boltzmann's constant.
