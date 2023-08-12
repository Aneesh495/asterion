# Independent reference comparison

MACE is a separately labeled reference implementation, pinned to mace-torch 0.3.14. It is not
imported into Asterion models or production accounting. The reference adapter exports the same
revealed structures as extended XYZ in physical units, fixes validation rows and retains exact
command arguments, logs, file hashes and process exits.

Create the comparison environment explicitly:

```sh
uv venv --python 3.12 .reference-venv
uv pip install --python .reference-venv/bin/python 'mace-torch==0.3.14' 'torch==2.8.0'
make reference CONFIG=configs/full.json RUN_ID=full-research
```

This command requires CUDA and uses 1000 training structures per molecule and seed. The separate
environment is not yet lock-verified or executed. Its compatibility, checkpoints and prediction
reproduction must be established before MA05 or a comparison result can pass. No pretrained
scores from the MACE publication are used as observed Asterion results.
