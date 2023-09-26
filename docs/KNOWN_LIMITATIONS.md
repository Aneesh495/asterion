# Known limitations and unmet requirements

The project has a working diagnostic path and does not meet the complete requested specification.
All full gates remain unverified or unmet. The preflight's blocked status must not be presented as
full acceptance. The default shipped configuration is the measured CPU diagnostic.

The available host is an Apple M2 MacBook Air with 16 GiB shared RAM and no CUDA. Full config requires
at least 16 GiB dedicated CUDA memory, float32 force-loss double-backward and a measured resource
pilot. No paid compute was provisioned. Accelerator training and replay tolerances are unvalidated.
GitHub authentication is invalid, there is no remote, and hosted CI/publication were not executed.

Implementation depth and authored semantic test populations remain below the requested delivery
requirements. Parameterized cases do not create additional authored test functions. The 164 passing
collected tests do not satisfy the 400 authored-test gate. No generated filler is counted.

The full acceptance command now orchestrates and assembles declared full stages after preflight.
The complete full path remains unexecuted and fails closed on missing requirements. Full gate checking exists but has no valid full positive bundle. Some full checks
still trust retained secondary/reference summaries more than their required raw model provenance.
Diagnostic control results cannot certify that unfinished path. Paired block uncertainty is a
library function, but the full learning gate has not yet integrated its required confidence interval.

The diagnostic trains one molecule at one budget. It does not execute the 72-run primary matrix,
the two-molecule three-seed acquisition matrix, full ablations, independent MACE retraining, or the
extended-system trajectories. Tiny two-epoch acquisition/ablation runs exercise code only. Near-zero
learning in those runs is retained. No superiority, calibrated generalization or label-efficiency
claim follows from them.

The author-split protocol contains correlated geometries. Audit descriptors expose close structures,
but do not prove independent conformations. Temporal shift exclusions can shorten the configured
shift population. Exact geometry-cluster independence is not established. MD22 units and hashes are
checked, but redistribution rights and its full study remain unverified.

Irrep feature contracts are now enforced at the actual interaction boundaries. Electronic levels
and trained component compositions are checked before learned inference. Current changes still
require fresh source-bound model execution.
The quadratic invariant readout is a particular higher-order path, not an arbitrary body-order
expansion. Finite radius graphs omit long-range interactions. Composition-specific offsets do not
establish elemental chemical transfer.

The retained learned trajectories cover 25 fs each and one initial geometry per seed/temperature.
They are not the million-step, 10,000-step-start campaign, an equilibrium structural analysis or a
matched long-horizon model comparison. No reference labels are known along generated trajectories.
Force error overlays therefore apply to held-out released geometries, not generated frames.
The single-model trajectory reports unavailable uncertainty. The viewer does not yet display a
full acquisition-round timeline or all committee/shift diagnostics.

The stress test measures repeated inference and cumulative process peak RSS. It does not isolate
allocator live memory, per-call peak GPU allocation or accelerator host transfer. Benchmarks were
run alongside other local tasks and are descriptive diagnostics. Dynamics throughput and cache
reuse scaling need dedicated matched campaigns. The reference environment remains unlocked.

Source fingerprints cover src, tests, configs, scripts, locks and Makefile. Documentation and CI
workflow bytes are not in that fingerprint. Evidence-only documentation changes preserve execution
identity, but changing scientific contracts in prose would require a new freeze. Paths to external
datasets in local bundles must be updated only by a new bundle generation after verified acquisition;
editing a completed manifest is not a relocation mechanism.
