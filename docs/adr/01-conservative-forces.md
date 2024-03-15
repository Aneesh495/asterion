# Derive forces from energy

Status: implemented. See [model derivation](../MODEL.md) and [force update](../WALKTHROUGHS.md).

Molecular dynamics needs forces in eV/angstrom and a consistent potential in eV. The production
model predicts a scalar energy and returns its negative coordinate gradient. Training fits both
energy and force residuals through that same scalar. The normalization shift is coordinate
independent; the energy scale must multiply the coordinate gradient before physical forces leave
the model. A missing sign or scale can look plausible while breaking integration.

A separate vector force head would avoid part of the derivative graph and reduce backward cost.
It would require an additional conservativity argument and could produce path-dependent work.
Numerical differences of a scalar could enforce a potential but require multiple forwards per
coordinate and introduce subtraction error. Neither alternative is used for production forces.
Central differences remain an independently expressed check.

Force-loss training retains coordinate derivatives and differentiates them with respect to model
parameters. This increases graph lifetime, memory and compute. Inference still needs coordinate
autograd even when parameter gradients are disabled. Blanket `no_grad` around inference is an
error. The physical Hessian action needs a further coordinate derivative and can be ill
conditioned near unsuitable geometries.

`test_genuine_force_loss_double_backward` and the retained derivative campaign exercise this
boundary. Detached forces, substituted units and changed checkpoint outputs must fail independent
verification. Conservative forces alone do not establish molecular accuracy or stable long
trajectories. Those claims require the separate held-out learning and dynamics populations.
