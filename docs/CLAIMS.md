# Claim-to-evidence map

Claims below are scoped to local CPU diagnostics. Raw paths are retained in this checkout's ignored
`runs/` and `evidence/`. They are deliberately not replaced by checked-in fabricated summaries.
The source digest and every figure input hash are in [figure provenance](figures/PROVENANCE.json).

| Behavioral claim | Source and independent check | Retained observation |
| --- | --- | --- |
| Forces derive from returned physical scalar energy | `models/potential.py`, `test_force_central_energy` | `runs/frozen-diagnostic-02/derivatives/derivatives.json` |
| Force loss changes original model parameters | `training/engine.py`, `test_genuine_force_loss_double_backward` | First mixed-gradient/history record and initial/selected tensor files for each run |
| The equivariant composition obeys tested O(3) transformations | `models/potential.py`, `evaluation/checks.py`, `test_o3_and_additivity` | 1200 retained diagnostic symmetry trials |
| Cached graph vectors stay differentiable after coordinate cloning | `graphs/radius.py`, `test_cached_edges_rebind_after_coordinate_clone` | Passing source-bound test invocation `runs/validation-02/tests.xml` |
| Graphs match independent all-pairs topology | `graphs/radius.py`, `evaluation/property_campaign.py` | 10,000 cases in `runs/validation-02/properties.json` |
| Unique label count is bounded and roles are separate | `data/ledger.py`, `test_label_oracle_access_boundary` | Every model's split, normalization identities and reveal chain; acquisition round ledgers |
| Real three-seed ethanol learning occurred | `training/engine.py`, saved-weight replay in `artifacts/verify.py` | Six selected checkpoints, 600 epoch rows, raw evaluation arrays and [held-out table](figures/heldout.csv) |
| Learned NVE steps obey recorded physical integration | `dynamics/integrator.py`, independently expressed SI `evaluation/replay.py` | Nine trajectory directories, 2250 completed steps and [energy table](figures/nve-drift.csv) |
| Invalid learned force terminates explicitly | `dynamics/detectors.py`, `test_force_detector_preserves_failed_start` | `runs/frozen-diagnostic-02/demo-failure/` and failure viewer |
| Recovery is deterministic for checked CPU cases | `training/checkpoint.py`, `dynamics/integrator.py`, `evaluation/recovery.py` | 50 simulation and five training interruption outcomes |
| Diagnostic corruption is rejected semantically | `artifacts/verify.py`, executed subprocess controls | `runs/frozen-diagnostic-02/controls.json`, 16 rejections, 13 rehashed controls and two positive exits |
| Batches were timed through coordinate/parameter derivatives | `evaluation/benchmark.py` | 490 raw samples in `runs/frozen-diagnostic-02/benchmark.json` |
| Figures describe actual outputs | Retained prediction/history/trajectory arrays | [CSV tables](FINDINGS.md) and hashed [provenance](figures/PROVENANCE.json) |

Unestablished claims include full sample efficiency, uncertainty-guided label advantage, reference
superiority, broad chemical transfer, long-horizon thermodynamics, full acceptance and hosted CI.
The requirements remain in [experiment plan](EXPERIMENT_PLAN.md) and [build status](BUILD_STATUS.md).
