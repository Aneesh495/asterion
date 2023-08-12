# Model derivation and inference contract

For directed edge j to i, the vector is r_j-r_i. A scalar radial network outputs tensor-product
weights as functions of its norm. Real spherical harmonics of order l transform with D_l(Q).
Parity is (-1)^l. An e3nn tensor product is an intertwiner, so its output carries the declared
irrep. Aggregation at each target commutes with rotations and permutations.

`EquivariantInteraction` performs an original composition of radial weights, tensor products,
constant neighbor normalization, equivariant linear mixing, parity-aware scalar gates and a
residual sum. The full hidden space contains repeated 0e, 1o and 2e blocks. Componentwise vector
normalization is never used. Squared whole-block norms and even scalar tensor contractions are
invariants. A scalar readout of these invariants produces atomic energies.

`E(R,Z) = scale * sum_i e_i(R,Z) + N * offset`

`F_i = -dE/dR_i`

The scale and offset are fitted only from revealed training labels. The offset is per molecule
with fixed composition; it is not a learned elemental chemical reference. Additivity holds for
disconnected graphs with the same declared offset convention. Local cutoffs omit long-range
electrostatics and dispersion. No universal transferability follows from this construction.

The quintic envelope `1-10x^3+15x^4-6x^5`, for x=r/cutoff below one, has zero first and second
derivatives at the cutoff. Beyond the cutoff it is zero. The radial basis uses sinc to avoid a
zero-distance division branch. Coincident atoms are rejected rather than hidden by a force clamp.
Neighbor normalization is a fixed expected degree, so edge birth does not renormalize every
existing message. The hard-cutoff variant is an explicitly discontinuous ablation.

Translation invariance implies sum_i F_i=0. O(3) scalar invariance implies zero total torque and
rotation/reflection covariance of forces. Tests include same-species permutations, changed batch
ordering, isolated atoms, collinear configurations and separated copies. Central energy
differences check force sign. Directional force differences check Hessian-vector products.
Relative derivative errors divide by the larger norm of the two quantities, with a 1e-8 floor.

Production inference accepts only supported isolated chemistry. `NonconservativeAblation` is a
separately labeled force-head experiment. It cannot supply a production conservative committee.
Tensor features carry irreps, atom axes, graph identities, precision and dimensionless units.
