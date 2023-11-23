# Capability register

`diagnostic passed` means a working mechanism with local executed evidence. It does not certify its
fixed full workload. `partial` identifies a connected but incomplete mechanism. `unverified` means
the requested execution or decisive evidence is missing. Test IDs refer to actual functions in tests.

| ID | Mechanism | Source | Check or artifact | Status |
| --- | --- | --- | --- | --- |
| M01 | Geometry, chemistry and units | `contracts/geometry.py`, `contracts/units.py` | Independent SI units and unsupported chemistry tests | diagnostic passed |
| M02 | Pinned data acquisition | `data/release.py`, `data/secondary.py` | Four primary hashes and MD22 author-file hash | diagnostic passed; full secondary study unverified |
| M03 | Splits and correlation audit | `data/splits.py` | Primary/shift splits, retained nearest-distance audit | partial; complete geometry-group independence unverified |
| M04 | Reveals and normalization | `data/ledger.py`, `training/engine.py` | Label access/reveal tests and verifier recomputation | diagnostic passed |
| M05 | Original batched radius graph | `graphs/radius.py` | 10,000 independent all-pairs cases | diagnostic passed |
| M06 | Smooth cutoff and cache | `graphs/radial.py`, `graphs/radius.py` | Cutoff derivatives and skin/batch invalidation tests | diagnostic passed |
| M07 | Irrep feature types | `representations/features.py` | Every model interaction retains representation and graph IDs | diagnostic passed |
| M08 | Scalar invariant potential | `models/potential.py` | Three original trained ethanol seeds | diagnostic passed |
| M09 | O(3) equivariant composition | `models/potential.py` | Three original trained seeds and symmetry trials | diagnostic passed |
| M10 | Quadratic higher-order readout | `models/potential.py` | Connected readout and no-higher-order integration ablation | diagnostic passed; full controlled ablation unverified |
| M11 | Physical negative energy gradient | `models/potential.py` | Central energy differences and independent unit replay | diagnostic passed |
| M12 | Force-loss double backward | `training/engine.py` | Nonzero mixed gradients and trained weight changes | diagnostic passed |
| M13 | Symmetry metamorphic checks | `evaluation/checks.py` | 1200 genuine retained model trials | diagnostic passed |
| M14 | Force, torque and additivity | `evaluation/checks.py` | Randomized original output trials | diagnostic passed |
| M15 | Force/Hessian differences | `evaluation/checks.py` | 1000 force and 100 Hessian checks | diagnostic passed |
| M16 | Three-seed original selection | `training/engine.py` | Six histories and validation-optimal selected checkpoints | diagnostic passed |
| M17 | Sample efficiency | `training/campaign.py` | One-budget ethanol; full matrix absent | unverified |
| M18 | Pinned independent reference | `training/reference.py` | Locked macOS environment and actual CPU training/export/evaluation | partial; full CUDA retrainings unverified |
| M19 | Committee/calibration | `uncertainty/committee.py` | 27 tiny member runs and held-out calibration arrays | diagnostic passed; useful ranking unestablished |
| M20 | Budgeted active acquisition | `acquisition/campaign.py`, `acquisition/policy.py` | Exact 8/16/32-label integration ledgers | diagnostic passed; required budgets/population unverified |
| M21 | Uniform/diversity controls | `acquisition/policy.py` | Same initial rows across three policies | diagnostic passed |
| M22 | Original Verlet | `dynamics/integrator.py` | Analytic checks and learned trajectory SI replay | diagnostic passed |
| M23 | NVE/NVT accounting | `dynamics/integrator.py` | Analytic NVT heat/noise and CPU recovery checks | diagnostic passed; learned NVT study unverified |
| M24 | Independent dynamics references | `evaluation/analytic_campaign.py` | 110 oscillator/LJ fixtures | diagnostic passed |
| M25 | Timestep refinement | `evaluation/analytic_campaign.py` | Three timesteps at equal physical horizon | diagnostic passed |
| M26 | Failure detectors | `dynamics/detectors.py` | Real extreme-force intervention and invalid cases | diagnostic passed |
| M27 | Independent physical replay | `evaluation/replay.py` | Complete retained diagnostic integration arithmetic | diagnostic passed |
| M28 | Atomic CPU recovery | `training/checkpoint.py`, `dynamics/integrator.py` | 55 interruption cases | diagnostic passed |
| M29 | Scientific viewer | `viewer/export.py`, `viewer/viewer.html` | Browser-checked real predictions and failure reason | partial; acquisition timeline absent |
| M30 | Raw resource/cost work | `evaluation/benchmark.py` | 490 samples; 30-minute stress | partial; full matched scaling unverified |
| M31 | Derivations and docs | `docs/` | Eight ADRs, ten editable/rendered DOT diagrams, four source walkthroughs | diagnostic documentation passed |
| M32 | Independent artifact verification | `artifacts/verify.py`, `artifacts/controls.py` | Valid diagnostic bundle and 16 actual corruption rejections | partial; full positive assembly/coverage absent |

Every full MA gate is separately tracked in [build status](BUILD_STATUS.md). A partial mechanism
cannot acquire a passed full status from a successful smoke workload.
