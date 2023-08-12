# Source walkthroughs

## Batched energy and force

`contracts/geometry.py:collate` validates chemistry and flattens positions to [sum atoms,3].
Pointers delimit graphs; labels remain [graphs] and [sum atoms,3].
`graphs/radius.py:RadiusGraph.topology` compares blocked detached coordinate pairs and returns
directed integer indices. `geometry_edges` recomputes differentiable edge vectors. The model
passes radial/spherical features through original interactions and sums atom energies by graph.
`models/potential.py:EnergyPotential.forward` returns the negative gradient of those physical
energies. `test_force_central_energy` independently differences energy values.

## Force-loss parameter update

`training/engine.py:train` obtains only revealed labels and fits their offsets. It sets
`create_graph=True` when evaluating forces, computes normalized energy and force losses,
and differentiates the force loss with respect to model parameters. The first update records a
nonzero mixed gradient. Weighted accumulation counts actual geometries in the final partial
batch. The optimizer updates parameters; validation chooses a checkpoint. `test_genuine_force_loss_double_backward`
checks parameter changes, and `training_recovery` compares uninterrupted and resumed tensor bytes.

## Physical integration

`dynamics/integrator.py:simulate` validates a candidate before calling the evaluator. NVE computes
the full drift from the previous velocity and force, evaluates the new force, then completes the
second kick. NVT instead performs BAOAB and records thermostat heat and exact Gaussian noise.
`dynamics/detectors.py:Detector` returns a termination reason. `evaluation/replay.py` reconstructs
the integration from saved arrays and SI. `test_simulation_atomic_resume` checks both ensembles.

## Acquisition round

`acquisition/campaign.py` initializes identical candidate rows for each policy. Three member seeds
train on one revealed set. `uncertainty/committee.py` predicts coordinates without reading labels
and calibrates on a disjoint role. `acquisition/policy.py:choose` combines score rank and greedy
descriptor novelty, or applies a matched control. Candidate scores and choices are saved before
`RevealLedger.reveal` publishes a new hash-chained round. Labels enter the next training round only
after this publication. `test_exact_budget_and_no_repeat` and `test_label_oracle_access_boundary`
exercise the information and budget contracts.
