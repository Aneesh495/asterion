# Disagreement and finite-pool acquisition

Three independently seeded conservative models predict energies and forces for the same coordinate
identities. Member energy standard deviation is in eV. For each force component, sample standard
deviation uses denominator two. The per-geometry score is the square root of the mean squared
component standard deviations over all 3N components, in eV/angstrom. Element count does not
silently turn a geometry into 3N independent uncertainty observations.

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

The tiny integration experiment's error correlations are low and its bounds wide. It did not
establish useful acquisition or shift-calibrated uncertainty. See [retained findings](FINDINGS.md).
The original full 100/300/1000-label, three-policy, three-seed, two-molecule experiment is pending.
