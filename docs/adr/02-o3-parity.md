# Preserve reflection parity

Status: implemented. See [representation conventions](../MODEL.md).

The molecular energy should be invariant under rotations and reflections, while coordinates and
forces transform as polar vectors. The equivariant model uses complete angular-momentum irreps
with polar parity `(-1)^l`. e3nn supplies harmonics and tensor-product intertwiners. Asterion owns
the feature identities, block composition, radial weighting, scalar gates, quadratic invariants
and energy readout. A dependency primitive is not evidence of the complete model's symmetry.

An SO(3)-only architecture could distinguish reflected geometries and would need a separate parity
constraint. Applying a scalar activation independently to vector components is cheaper to write
but generally destroys covariance. Scalar invariant features are simpler and cheaper; they remain
the original controlled baseline, with separate training and held-out evaluation.

Whole-irrep gates preserve transformation rules, but tensor products add paths, radial outputs
and intermediate activations. Higher angular order changes computational cost and memory. The
fixed ablations remove vector features or higher-order readout while preserving the remaining
training contract, so any gain or loss is an observed result rather than a presumed benefit.

The representation wrapper binds tensors to their irrep specification and graph identity. Width
or graph mismatches fail before interaction. Tests also transform energies, forces, translations,
atom permutations and reflected coordinates using independent metamorphic identities. Finite
precision imposes tolerances; exact symmetry in the model equations does not imply bitwise
identical accelerator outputs. The current CPU diagnostics do not certify accelerator symmetry
or the requested matched scaling experiment.
