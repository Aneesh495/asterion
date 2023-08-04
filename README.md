# Asterion

Asterion investigates whether geometric tensor representations improve force learning and
trajectory reliability under fixed molecular label budgets. Its original potentials predict
scalar energies. Coordinate gradients supply conservative forces to an original dynamics engine.

The supported domain is isolated neutral molecules containing H, C, N and O. Labels from different
electronic structure methods belong to separate experiments. The full research campaign is pending.
See [build status](docs/BUILD_STATUS.md) for execution evidence and outstanding gates.

## Development

```sh
make bootstrap
make doctor
make test
```

Python 3.12 is required. CPU float64 is the reference precision. Research execution requires
a compatible accelerator with force-loss double-backward support. Dataset files and tensor
checkpoints stay outside Git.
