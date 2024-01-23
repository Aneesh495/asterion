# Physical integration and verification

Let R be positions in angstrom, V velocities in angstrom/fs, m masses in amu and F forces in
eV/angstrom. Define c = 0.009648533202185 angstrom/fs squared per (eV/angstrom/amu).
Then a = c F/m, with mass division per atom. The energy conversion makes kinetic energy
K = sum_i m_i |V_i| squared / (2c) in eV. [Unit derivation](UNITS.md) repeats this through SI.

For NVE, velocity Verlet implements

```text
R(n+1) = R(n) + dt V(n) + dt squared a(n)/2
F(n+1) = -gradient E(R(n+1))
V(n+1) = V(n) + dt [a(n) + a(n+1)]/2
```

The learned energy is evaluated at the exact candidate positions used in the second kick.
Topology-cache reuse changes indices only, never coordinate vectors. Detectors reject a candidate
before force evaluation and reject a bad force result before publishing a completed step.
The numerical integrator does not promise exact energy conservation. On smooth forces it has
second-order global coordinate accuracy at a matched fixed physical horizon. Decreasing dt and
decreasing the horizon would not test that claim.

NVT uses the BAOAB Langevin composition: half kick B, half drift A, Ornstein-Uhlenbeck O, half drift
A and half kick B. In O, q = exp(-friction dt), and every velocity component changes to
q V + sqrt[(1-q squared) c k_B T / m] xi, with independent unit Gaussian xi. Heat is the kinetic
energy change during O, before the remaining deterministic operations. The exact xi arrays are
retained. The mechanical accounting compares E+K and E+K-heat separately. Temperature is a kinetic
diagnostic; a thermostat-controlled temperature does not prove force accuracy or NVE conservation.

Initial velocities are sampled from the declared temperature, center-of-mass motion is removed,
and they are rescaled to 3N-3 kinetic degrees of freedom. No rotational constraint is removed.
The simulation policy tracks distance/bond/force/uncertainty heuristics relative to its initial
geometry. These are finite-run validity criteria, not a reaction classifier.

[Independent replay](../src/asterion/evaluation/replay.py) expresses acceleration and kinetic energy
through SI constants rather than importing the production integrator. It reconstructs every
completed kick/drift and heat event from arrays. Model-weight probes separately compare retained
forces/energies to the saved checkpoint. Agreement of those checks establishes recorded execution;
it does not supply independent electronic structure labels along generated trajectories.

[Analytic campaign](../src/asterion/evaluation/analytic_campaign.py) checks oscillator trajectories
against the analytic sine/cosine solution and LJ trajectories against an independently expressed
SciPy reference ODE. Three timesteps share one physical horizon. The retained campaign has 110
fixtures. [Dynamics tests](../tests/test_dynamics.py) additionally check thermal units, exact CPU
NVE/NVT continuation, second-order refinement and failed-start retention.

Atomic snapshots publish immutable arrays and metadata before updating a pointer. Simulation RNG,
state and completed frames resume from that pointer. Rebuilding a learned neighbor cache after
recovery may change diagnostic rebuild counts without changing the force. CPU analytic recovery
is exact in retained arrays; accelerator continuation remains an unmeasured numerical contract.

The separate learned study uses selected conservative weights and the first retained evaluation
geometry for each declared molecule/seed. Full execution runs a 300 K NVT path for 1000 steps at
0.1 fs. Three NVE paths use the same initial state and timesteps 0.1, 0.05 and 0.025 fs, with
1000, 2000 and 4000 requested steps respectively. Their completed horizons must match before
coordinate differences are compared. Failed refinements remain recorded with comparison
unavailable. The observed coarse/middle difference ratio is reported without an asymptotic-order
claim. CPU float64 is explicit for this physical-check study.

Structural reports exclude the initial frame. Bond references use the detector's declared
initial-distance heuristic, with species-pair histograms on fixed 0.05-angstrom bins from zero to
six angstrom. Out-of-range observations remain explicit counts. Means and sample deviations use
all retained samples, including overflow values. The time series is correlated; these deviations
are distributions of a finite path, not independent-frame standard errors or equilibrium estimates.

Temperature autocorrelation uses the centered recorded series and up to 200 lags. The positive
sequence before its first nonpositive value supplies a finite-path integrated-time diagnostic.
Constant and short series report this quantity unavailable. Center-of-mass displacement and
internal angular momentum changes are derived from retained masses, coordinates and velocities.
NVT drift subtracts the actual recorded thermostat heat. These checks cannot supply electronic
structure truth along generated coordinates.
