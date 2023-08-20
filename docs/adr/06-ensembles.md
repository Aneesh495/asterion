# Separate NVE from thermostatted NVT

Status: implemented.

A thermostat can hide potential or integrator energy drift. NVE velocity Verlet and NVT BAOAB have different evidence contracts. NVT records Gaussian noise and thermostat heat so independent replay can account for energy exchange. Finite-run temperature sampling is not an equilibrium proof.

The consequences are exercised through the standard model, training, simulation or verifier path. Full scientific acceptance remains a separate execution obligation.
