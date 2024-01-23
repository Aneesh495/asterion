# Frozen experiment contracts

The primary endpoint is geometry-averaged Cartesian force MAE in eV/angstrom. Energy MAE,
force RMSE, net force, torque, derivative error, physical energy drift and trajectory completion
are separate endpoints. Error bars resample declared geometry blocks, not atom rows. A positive
paired improvement interval is required for the demonstrable-learning gate.

Full primary experiments use aspirin, ethanol, malonaldehyde and salicylic; seeds 1,2,3;
label budgets 100,300,1000; original invariant and equivariant models. Labels and split membership
are identical within each comparison. Models start from random initialization. Checkpoints are
selected by minimum validation force MAE. Final evaluation has at least 5000 geometries per molecule.
Reference MACE retraining uses the 1000-label budget in a separate environment.

The guarded comparison is a separate original training task at 1000 labels. For each primary
molecule, both architectures and all three seeds train on the first 45 percent of original-time
ordering. Validation, calibration and evaluation occupy the next 15, next 15 and final 25 percent.
Each block excludes an additional 100 original-index units at each boundary. Evaluation retains
5000 geometries. These 24 models are additional to the 72 author-split primary models. Checkpoints,
normalizers, labels and selection remain separate. `make guarded-shift` executes this path.

The diagnostic profile uses ethanol with the same three seeds, 100 labels, 100 epochs and 256
held-out geometries per seed. It answers whether the integrated code learns and runs. It does not
answer the full cross-molecule sample-efficiency hypothesis. Pilot test inspection is retrospective.
No model tuning based on these results is claimed as untouched final evaluation.

Acquisition uses shared initial rows and three policies: uniform, diversity, and disagreement plus
diversity. Three independently seeded models train on the same revealed rows. Calibration has its
own held-out role. The finite pool oracle is the released dataset, not new electronic-structure
calculation. Committee variance is an empirical score, not a guarantee of epistemic uncertainty.

Ablations vary angular order, quadratic invariant interaction, cutoff smoothness, force-loss
training and acquisition policy. Energy-only training still reports force error from gradients.
A nonconservative force head is a separately labeled experiment. The full ablation and extended
system populations remain required even when a diagnostic comparison looks favorable.

NVE trajectories use held-out initial states at 100,300,500 K. The full profile declares 10,000-step
starts and requires one million valid executed steps plus at least 90% core-domain completion.
Failures remain in the denominator. No replacement starts are currently authorized by the frozen
profile. NVT runs are separate and account for heat exchanged with a stochastic thermostat.
The additional learned study fixes aspirin/ethanol, all three seeds and 300 K. Each selected model
has one 100 fs NVT path and three NVE refinements at the same 100 fs horizon. Completed primary
NVE steps are counted separately from these additional study steps. Short CPU study cases test
the code path and cannot establish thermostat equilibration or molecular stability.

Full gates require 400 authored semantic tests, 10,000 generated contract cases, 1200 symmetry
trials, 1000 force checks, 100 Hessian checks, 100 analytic fixtures at three timesteps,
50 recovery cases, 100 failure cases, a 30-minute memory stress and real verifier controls.
`make acceptance` never replaces these with the diagnostic profile.

Paired learning uncertainty uses original temporal index blocks of span 1000, with 2000
block-bootstrap repetitions and seed 491. The point estimate remains geometry weighted;
resampled block totals divide by resampled geometry counts. Positive lower bounds are required
against zero force and the untrained model for the full learning gate. Blocks are a correlation
heuristic and do not establish independent molecular conformations. The primary model's temporal
slice separately oversamples permitted candidates, excludes every primary role and retains the
exact declared count. That slice has no guard relative to its randomly sampled training labels;
it cannot substitute for the separately trained guarded comparison.
