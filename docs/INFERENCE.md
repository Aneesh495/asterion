# Inference contract

`Geometry(numbers, positions, identity, chemistry)` requires integer atomic numbers [N], finite
float64 positions [N,3] in angstrom and a stable identity. N is positive. The electronic label level
and isolated, neutral singlet chemistry are explicit. Unsupported elements, charge, multiplicity,
periodic cells, nonfinite values and nearly coincident atoms are rejected. An identity names a
released observation or an explicitly declared synthetic fixture.

`collate` creates numbers [sum N], positions [sum N,3], graph IDs [sum N] and pointers [G+1].
`EnergyPotential.forward(batch, create_graph=False, edges=None)` returns energy [G] in eV and force
[sum N,3] in eV/angstrom. Forces derive from the exact returned scalar energy. Inference retains
coordinate autograd and explicitly enables it inside `forward`. Use `create_graph` for force-loss
parameter gradients or Hessians. Detach outputs only after derivatives are computed.

Graphs contain directed source-to-target edges within one geometry and inside the cutoff. No edge
crosses a batch boundary. Cached indices are rebuilt after motion exceeds half the skin. Edge
vectors always use current differentiable positions. Supplying indices built on detached positions
does not detach energy from the current coordinate tensor. Empty neighborhoods remain supported.

`load_checkpoint` reads JSON and safetensors, validates immutable tensor identities, reconstructs
the original architecture and restores weights. Model precision follows the saved checkpoint;
callers collate inputs to the actual parameter dtype and device. Dataset electronic levels cannot
be inferred from atomic numbers and must be checked by the application at the data boundary.

`LearnedEvaluator` owns a model and neighbor cache, returns `ForceResult`, and converts forces to
NumPy only at the integration boundary. `CommitteeEvaluator` averages three conservative member
energies and forces. Its force score is the root mean square of componentwise sample standard
deviations, so it carries eV/angstrom units. Averaging gradients preserves conservativity.

`ValidityPolicy` sets minimum distance, maximum force, bond ratio and optional uncertainty limit.
Geometric checks happen before a force call. Result checks reject nonfinite energy/force, excessive
forces and excessive uncertainty. A required but unavailable uncertainty score is an error, not
a zero-risk observation. No oracle replaces a rejected learned force. Failed candidate arrays and
the reason remain in the run. The policies are heuristics, not chemically complete safety proofs.

For direct use, execute within the locked environment:

```python
import numpy as np
from asterion.contracts.geometry import Geometry, collate
from asterion.models.potential import ModelConfig, make_model

geometry = Geometry(np.array([6, 1]), np.array([[0., 0., 0.], [1.1, 0., 0.]]), "fixture")
model = make_model(ModelConfig(width=4, depth=1, radial=4)).double()
energy, force = model(collate([geometry]))
```

This example exercises shapes and units using untrained weights. Actual learning and dynamics
use the [reproducibility commands](REPRODUCIBILITY.md) and selected original checkpoints.
