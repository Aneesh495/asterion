# Disagreement and finite-pool acquisition

Three independently seeded conservative models predict energies and forces for the same coordinate
identities. Member energy standard deviation is in eV. For each force component, sample standard
deviation uses denominator two. The per-geometry score is the square root of the mean squared
component standard deviations over all 3N components, in eV/angstrom. Element count does not
silently turn a geometry into 3N independent uncertainty observations.

Moment calculation promotes finite member outputs to float64 and scales each component before
summing or squaring. Per-geometry RMS uses another scale to avoid overflow in squared deviations.
Ordinary inputs reproduce direct sample moments. A mathematically unrepresentable deviation,
calibration ratio or final bound raises a contract error; no infinity or clipped surrogate enters
the production disagreement record.

Committee mean force is the negative derivative of committee mean energy. A nonconservative
force-head ablation cannot join the production committee. Sharing training labels is intentional;
different member seeds do not make correlated models statistically independent estimators of a
Bayesian posterior. Their disagreement is an empirical quantity.

Calibration uses a disjoint allowed geometry role. For geometry error e and score s, form
e/max(s,floor), with floor = max(0.01 median(s), 1e-8). The sorted ratio rank is
min(n, ceil[(n+1)(1-alpha)]). The selected ratio multiplies max(s,floor) at inference. The retained
alpha is 0.1. A finite-sample exchangeability argument would require exchangeable calibration and
evaluation units. Correlated MD frames and temporal shift do not justify that premise here.
Coverage and rank correlation are measured rather than guaranteed.

Acquisition is label-bounded. Released labels serve as a controlled finite-pool oracle. Coordinates
and invariant species-pair distance descriptors may be inspected before reveal; energy and forces
may not. Every policy starts from the same frozen first-round rows. Uniform selection samples the
remaining pool; diversity-only chooses descriptor novelty; uncertainty-plus-diversity combines
ranked disagreement with novelty. Scores, candidate rows and selected rows are retained before
the immutable reveal round is published. A repeated frame does not count as a new label.

The labels used for fitting, checkpoint selection, calibration and final evaluation occupy distinct
roles. A Python interface enforces ordinary call access but is not an adversarial sandbox. Round
lineage connects split digest, reveal digest, member weights and calibration identities. Full
analysis must compare matched final evaluation units, training cost, diversity and failure rate.

Each round records monotonic clock observations for member training, selection/reveal, calibration
and committee evaluation, and candidate inference. Overall round time includes checkpoint and
prediction writes and per-member evaluation. Shared descriptor preprocessing and the initial
reveal precede this interval; cost-summary and final-pairing publication follow it. Those exclusions
are explicit. Optimizer counts, processed examples and epoch durations are reconstructed from
retained histories. This is measured finite-pool work, not a measured electronic-structure oracle
cost or isolated hardware throughput claim.

Selected-geometry diversity uses species-pair coordinate descriptors standardized over the
permitted candidate pool. Reports retain mean nearest and mean pairwise descriptor distance,
with source coordinate reconstruction. Final policy comparisons use identical evaluation
identities and geometry-weighted errors. The paired temporal-block bootstrap has span 1000,
2000 draws and seed 491; a single block reports the interval unavailable. A negative or
inconclusive acquisition result remains a result.

The tiny integration experiment's error correlations are low and its bounds wide. It did not
establish useful acquisition or shift-calibrated uncertainty. See [retained findings](FINDINGS.md).
The original full 100/300/1000-label, three-policy, three-seed, two-molecule experiment is pending.

Committee members must load original trained checkpoints. The constructor checks actual seed and
checkpoint identities, distinct model objects, the physical task and executed parameter/buffer
fingerprints. Precision conversion or weight updates after loading invalidate that identity.
Inference checks the fingerprints before and after evaluation. Member list ownership is captured
so a caller cannot remove a member by changing the original list.

The loaded model's physical configuration is bound separately. Changing an energy scale, offset,
cutoff or label level after checkpoint loading cannot keep the old committee identity merely
because tensor values are unchanged. Configuration hashes are checked at construction and before
and after every inference call.

Retained training and validation identities are automatically excluded from calibration. Allowed
calibration identities are the intersection of the three saved calibration roles. A caller cannot
admit final evaluation labels by omitting a forbidden-row argument. Geometry identities must be
unique within each inference/calibration population. Bounds require a finite nonnegative vector;
zero bounds from a perfectly matched calibration sample remain available observations.

Trajectory records distinguish unavailable scores, raw force disagreement and calibrated force-MAE
bounds. Integer quantity codes and boolean availability are retained with every frame. Numeric zero
does not imply unavailability. Independent replay rejects contradictory codes, changing quantities
or metadata labels. The viewer uses that contract and labels calibrated bounds separately from
sample disagreement. Force overlays share one display scale; generated frames have no reference
labels, so reference/error overlays are disabled there.

Each acquisition trial additionally freezes a temporal suffix drawn from the last quarter of the
author's original trajectory order. It excludes the complete primary train pool, validation,
calibration and held-out evaluation roles. All policies and rounds use the same frozen rows and
the same geometry denominator as primary evaluation. The three saved members retain separate
predictions, physical author labels, coordinates and checkpoint probes for that suffix.
Calibration is reused unchanged; no shifted labels fit the multiplier or acquisition decisions.
Ranking, empirical coverage and mean bounds are independently recomputed for both conditions.
This is a disjoint temporal slice of a randomly sampled primary learner, not a temporally guarded
training comparison or a guarantee of out-of-distribution examples. The separate guarded study
retains its stronger information boundary.
