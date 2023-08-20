# Derive forces from energy

Status: implemented.

A separate force head reduces derivative cost but does not enforce a scalar potential. A coordinate gradient of the same physical energy is the production force. This adds mixed derivatives during training and requires retained graph memory. Finite energy differences test sign and scale; detached force tensors must fail verification.

The consequences are exercised through the standard model, training, simulation or verifier path. Full scientific acceptance remains a separate execution obligation.
