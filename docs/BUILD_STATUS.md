# Build status

Phase: learned thermostat/refinement integration and study analysis. Full delivery remains incomplete.

The current source passes 616 tests, covering 428 authored test functions, and Ruff/mypy.
`runs/replay-validation-17/tests.xml` records the full invocation. The 400-authored-test gate
is met. Private implementation accounting remains outside the repository. No workload requirement
was lowered.

The separate macOS MACE environment now matches its 44 hashed dependency pins. A real two-epoch
CPU compatibility run completed training, export and eight-geometry evaluation with the declared
reference architecture. Both commands exited zero; the raw journal independently shows eight
optimizer updates and 32 processed examples. The first export attempt failed because its model
directory was absent and is retained. Directory creation and virtual-interpreter symlink handling
are fixed. The Linux CUDA lock is resolved; its runtime and full retrainings remain unverified.

## Current mechanisms

Full acceptance executes the fixed stage plan through an exclusive coordinator with bounded
subprocesses, retained exits/logs/output hashes and completed-stage continuation. Training resumes
immutable optimizer generations. Interrupted stages without a continuation path retain partial
observations and restart. SIGINT and SIGTERM stop the owned process group. A live orphan after
SIGKILL is not duplicated. New training generations contain their own history and selection;
recovery archives future generations before restoring the resume epoch. Tests inject interruption
between checkpoint, selection and latest publication, and during recovery archiving.

Model interactions use irrep/graph-tagged tensor features. Models retain electronic level and
composition. Inference validates active connected components and message allocation budgets.
The MACE adapter exports labels, calls pinned training/evaluation, parses bounded XYZ and retains
physical predictions. Verification counts actual reference batches and epochs and checks every
exported input role against source labels. Dataset verification now also checks the pinned author
release, file IDs, sizes, MD5s, electronic level and license. Full reference retraining is unexecuted.

Independent study checking replays saved original weights, checks author labels, reconstructs
committee mean/disagreement/calibration and greedy acquisition choices, and replays secondary
trajectories. Paired learning intervals use geometry-weighted temporal blocks. Candidate
verification cannot grant release status; final full verification still requires actual corruption
controls and every scientific workload.

Committee construction now requires three saved trained checkpoints with matching seed and
identity records, distinct model objects and executed weight fingerprints. It rejects changed
weights or precision and enforces the intersection of retained calibration roles automatically.
Training and final evaluation labels cannot be admitted by omitting a caller forbidden list.
Calibration requires finite nonnegative scores/errors, distinct geometry/member identities and
valid fitted parameters. Tests use three genuinely trained original fixture models.

Acquisition-integration-09 executed and independently verified all nine rounds with the stricter
committee contracts. The viewer now records and displays uncertainty quantity and availability,
including available zero bounds. Raw replay rejects mislabeling. Actual browser checks exercised
the shared force overlays, playback and a retained uncertainty-threshold failure at zero of 16
steps. Reference/error overlays are disabled for generated coordinates. Console errors were absent.
The browser screenshot and real committee trajectories are retained in `runs/viewer-10`.

Managed stages retain every published checkpoint generation, raw predictions, study observations,
process log and failed attempt. Independent receipts check order, actual exits, continuation and
relocated output hashes. `managed-integration-13` exercised four real CLI stages, two original
models, 32 held-out predictions, eight learned steps and 32 saved-weight probes. All 16 actual
corruption controls rejected, including 13 with recomputed manifests; positive verification passed
before and after. Its execution commit is ece5cf7 and source digest is
`c33ff1f6026fd8d99d5d1ab1bbe0cbf409c725313ce093025ec0f828b40ef8d5`.
It is a two-epoch integration check, not useful-learning or full acceptance evidence.

Runtime measurement now includes host/device copies, Hessian products and actual learned dynamics.
The checker derives the full scaling matrix, replays saved outputs and SI trajectories, reconstructs
velocity/cache traces and checks that throughput uses completed work. Both original architectures
pass 48 new semantic cases. `runtime-integration-14` retained all 740 raw samples and ten actual
100-step dynamics starts. Independent saved-weight and SI replay verified 1000 completed steps.
Its execution commit is da6c029; the measured checkpoint comes from the two-epoch managed run.
This is a CPU runtime observation, not a useful-learning or molecular-stability result. CUDA timing
remains unexecuted.

Rejected initial geometry, invalid force/energy and initial evaluation exceptions now retain
`failure.npz` and an explicit rejected-initial record with zero valid frames. Nonfinite outputs
remain raw observations, never valid trajectory values. Independent checking reconstructs failure
reasons and can replay the rejection using saved weights. The viewer represents unavailable
force/energy honestly. Actual killed-coordinator tests preserve unknown exit/end/duration, logs
and successful continuation. A live orphan or unknown launch identity refuses duplicate work.

Committee moments use scaled float64 arithmetic; finite large inputs do not overflow intermediate
squares. Unrepresentable moments, calibration ratios and bounds fail explicitly. Dataset loading
checks ZIP populations, NPY headers, native types, shapes and byte budgets before reading array
bodies. Fractional elements/indices cannot be silently cast. All four author primary files and
the 85109-frame, 42-atom MD22 file loaded successfully through the new bounded reader. This parser
integration is retained in `runs/data-compatibility-15`.

Separate learned NVT and matched-horizon refinement studies are now connected to simulation and
bundle verification. The full protocol specifies two molecules, three seeds, a 1000-step NVT
path and three NVE refinements at 0.1/0.05/0.025 fs sharing a 100 fs horizon. These studies retain
temperature time correlations, species-pair bond distributions, center-of-mass/angular momentum
changes and heat-adjusted drift. Initial frames are excluded from structural samples. The checker
reconstructs each report, source geometry, initial velocity and saved-weight trajectory, and
rejects missing/shortened studies. CPU fixture tests pass; fresh learned-model integration is pending.

`managed-integration-16` executed all four CLI stages and retained the learned NVT/refinement
paths, but independent verification failed on angular-momentum cancellation. The source's
subtractions differed by about 6.5e-49 kg m²/s on a retained momentum scale near 1.9e-33 kg m²/s.
The failed invocation and original observations remain unchanged. The checker now derives a
conditioning-aware float64 comparison scale and requires a boolean resolution claim. Genuine
original-model tests cover cancellation and still reject an invented 1e-40 kg m²/s change.
Fresh source-bound integration remains required.

`managed-integration-16b` independently verified the corrected studies and all four managed CLI
stages at execution commit e68a71f. Its digest is
`86e00ab28a3758c3de676d6252b4f1114708d970873ddedd3033d859904ccf0d`.
The raw NVT and refinement paths total 32 additional steps; the primary path has eight. These
short observations exercise source/geometry/velocity/weight/report wiring. They do not establish
equilibration, useful molecular learning or full acceptance.

Replay now checks completion status against actual requested work, finite stochastic/policy
configuration, all Cartesian/observation axes, bounded array bodies, seeded NVT noise, per-frame
geometry/force validity and the integrator's failed candidate. Exponent-overflow JSON cannot
disable limits or conceal nonfinite arithmetic. A valid trajectory prefix cannot claim completion.

The rejected-initial viewer was checked in the actual browser at execution commit 3569e12.
It displays unavailable energy/force, disables force overlays and shows the retained collision
reason at zero of 16 steps. No console errors or horizontal overflow were observed. The screenshot
and DOM observations are retained under `runs/viewer-10/initial-rejection-15`.

## Executed checks

| Run | Observed result | Scope |
| --- | --- | --- |
| `integration-05` | Two original models, 16 held-out geometries each, eight learned steps, 16 saved-weight probes and 16 rejected controls | Two epochs at execution commit 144cd5e; integration only |
| `acquisition-integration-05` | Nine policy/budget rounds and 27 original committee members independently checked | Shared initial pool, calibration and decisions reconstructed; no policy advantage established |
| `secondary-integration-05` | Original MD22 model, 64 held-out geometries, eight probes and two trajectories totaling 32 steps | Separate electronic level and composition; two-epoch integration only |
| `recovery-validation-06` | 295 tests pass, including actual SIGINT/SIGTERM child termination and exact CPU training recovery | Current recovery source; no CUDA recovery claim |
| `integration-08` | Two original models, primary and temporal-slice predictions, 32 weight probes, eight steps and 16 rejected controls | Frozen execution commit 83d8df4; two-epoch integration only |
| `guarded-integration-08` | Two separately trained temporal-block models and 16 saved-weight probes | Frozen execution commit 83d8df4; small guarded comparison |
| `ablation-integration-08` | Five genuine interventions and 40 saved-weight probes independently verified | Frozen execution commit 83d8df4; no useful two-epoch learning claim |

Source-05 has digest `5b76e3a555780cd63159b89db737a2207da6a727b887ab830a373d4546163cf1`.
The small runs barely change force error. They establish wiring and artifact checks, not useful
molecular accuracy. Current recovery edits require fresh execution before claiming a new bundle.

Historical `frozen-diagnostic-03` belongs to execution commit
`6a99cd8210752e2e32bd8bebad75057479eb964f`, source digest
`7ca0c4acc802f7828c449d47db9f440be7860e274eaf8f1ed5257f6371d9a364`.
Its six 100-epoch models, 2250 learned steps and 16 negative controls verified. Equivariant force
MAE was 0.111071, 0.131622 and 0.129603 eV/angstrom, 84.9 to 87.1 percent below zero force.
The 30-minute stress completed 187406 calls with 417398784 bytes sampled peak RSS and zero
post-warmup growth. The exact checkout is retained at sibling `asterion-diagnostic-source`.
Its bundle certifies that checkout only. Current executable edits invalidate reuse for new source.

`learned-committee-03` gave held-out force MAE 0.097912 eV/angstrom, error-ranking Spearman
0.6154 and observed coverage 0.9141 on 256 geometries. Its 252-geometry temporal slice gave MAE
0.094420, ranking 0.6255 and coverage 0.9246. A calibrated-uncertainty trajectory completed
100 steps. These observations do not establish acquisition advantage or chemical transfer.
Older diagnostics, negative results and raw records remain retained in ignored local storage.

## Outstanding gates

Full execution is blocked on this Apple M2 host: CUDA is unavailable and the reference CUDA runtime is unavailable. No full 72-run primary campaign, 12 MACE retrainings, million-step dynamics,
full acquisition/ablation population or MD22 long-horizon study has run. Full positive assembly,
clean installations and package release have not been exercised against current source.

Guarded comparison and temporal-slice raw validation, saved-checkpoint history/normalizer checks
and raw ablation verification are implemented. The guarded CPU integration verified two saved models;
five original ablation fixtures replay saved weights and reject altered summaries.
Complete gate/source/command coverage, reference compatibility and resource pilot, acquisition viewer paths, uncertainty
validity checks and substantive numerical tests remain to finish. A temporal slice of an
arbitrarily sampled training pool does not establish a temporally guarded training comparison.

No GitHub remote or hosted publication exists. The last account probe found an invalid Aneesh495
keyring token. No paid resources were provisioned. Public publication needs a working account
and direct user authorization; the attached brief alone does not supply it.

Next: measure current runtime, finish initial failure retention and crash receipts, acquisition costs,
viewer paths and resource pilot, then freeze source for new diagnostic and clean builds.
On suitable authorized hardware, run the unchanged full workload and verify its actual outputs.
