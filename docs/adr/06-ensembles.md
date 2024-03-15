# Separate NVE from thermostatted NVT

Status: implemented. See [dynamics](../DYNAMICS.md) and [structural studies](../BENCHMARKS.md).

A thermostat exchanges energy with a system and can mask drift that would be visible in
microcanonical integration. Asterion therefore treats NVE velocity Verlet and NVT BAOAB as separate
protocols. NVE tracks total physical energy drift. NVT retains every Gaussian noise input and
thermostat heat exchange so independent SI replay can account for stochastic work.

Using one thermostatted trajectory for all stability claims would be convenient but would not
measure the stated NVE property. A deterministic velocity rescaling thermostat is cheaper to
implement yet has a different sampling contract. Delegating all integration to an external
molecular dynamics package would remove the original integrator obligation. External knowledge
informs the equations; production integration and decisive replay arithmetic are separate code.

All modes include physical masses and unit conversions. The NVT noise record adds storage, and
retaining every valid frame adds bandwidth and disk cost. Temperature is an instantaneous kinetic
quantity; a finite run's mean does not prove equilibrium or correct configurational sampling.
Structural distributions are reported against the available reference observations with explicit
denominators and finite-run limitations.

Timestep refinement compares equal physical horizons, rather than equal step counts. Detectors
retain failed candidate frames and explicit reasons; a thermostat cannot turn a failed NVE start
into a completed NVE observation. Harmonic and Lennard-Jones fixtures check integration against
independent analytic mechanics. Learned trajectories then require actual checkpoint replay and
held-out physical checks. Passing those local fixtures does not establish the million-step learned
campaign, the long MD22 study or accelerator performance.
