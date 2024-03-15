# Use a C2 compact envelope

Status: implemented. See [cutoff and cache contract](../MODEL.md).

A radius graph changes discretely when an interatomic distance crosses its cutoff. A hard energy
contribution at that boundary can create jumps in energy and force. Production radial features
use a quintic compact envelope whose value, first derivative and second derivative vanish at the
cutoff. The model uses fixed expected-degree normalization so a changing neighbor count does not
rescale all surviving messages discontinuously.

A hard cutoff costs less arithmetic and is retained as an explicit architectural ablation. A
cosine taper removes the first-derivative discontinuity but lacks the chosen second-derivative
boundary contract. An unbounded graph avoids a radius boundary while introducing quadratic edge
storage and interactions outside the declared local approximation. The compact model does not
supply an explicit long-range electrostatic or dispersion correction.

Neighbor reuse stores candidate identities inside cutoff plus skin. It does not cache detached
physical distances or coordinate derivatives. Current distances are recomputed from current
coordinates, and the envelope removes inactive candidates. Movement, atom ordering, batch
ownership and configuration invalidate the candidate cache under its documented lifetime rule.
The skin increases stored edges and can consume memory without improving reuse on large motion.

Independent all-pairs graph cases check edge identities. Boundary finite differences check the
radial derivatives; cache tests move atoms and change batches. Collisions, nonfinite coordinates
and unsupported resource demand fail explicitly. Smooth boundaries remove one integration
failure source, but large timesteps, extrapolated forces and omitted physics can still terminate
a trajectory. No stability claim follows from the taper alone.
