# Calibrate an empirical three-model committee

Status: implemented. See [uncertainty contract](../UNCERTAINTY.md).

Force disagreement is useful only if it belongs to distinct, actually trained predictors. A
committee requires three distinct saved checkpoints with matching physical task, chemistry and
saved split roles. State fingerprints reject three aliases of the same weights. Component sample
variance is reduced to a geometry score, and a held-out ratio calibration maps that score to an
empirical force-error bound under the declared calibration procedure.

A single model's force magnitude is cheap but measures neither disagreement nor error. Dropout
samples require a different training and stochastic inference contract. Larger ensembles may
improve robustness but multiply training, inference and derivative cost. Three original seeded
members implement the fixed campaign obligation while keeping those costs visible.

Calibration uses the intersected permitted calibration identities. It cannot inspect candidate or
final evaluation labels. Acquisition uses the resulting score and declared diversity descriptor;
paired policies share the initial pool and evaluation units. Saved raw error, score and bound
arrays support independently expressed rank correlation, observed coverage and mean-bound checks.
Constant rankings are unavailable rather than falsely assigned a meaningful correlation.

Models can share errors, especially outside their training distribution. The bound is empirical
and does not claim universal epistemic validity. Shift coverage must be measured separately.
Trajectories distinguish unavailable uncertainty from available zero disagreement or a zero bound.
Nonfinite scores, repeated checkpoints, altered seed lineage and changed calibration rows fail.
Tiny local committee runs establish this wiring but cannot establish acquisition advantage or
useful ranking at the required full label budgets.
