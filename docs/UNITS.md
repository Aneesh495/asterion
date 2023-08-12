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
