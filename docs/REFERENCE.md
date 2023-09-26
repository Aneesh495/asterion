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

Labels use REF_energy and REF_forces to avoid ASE reserved-key relocation. After training,
the named standard final export is evaluated with the pinned eval_configs CLI. Retained XYZ
has PRED fields and original identities. A bounded original parser saves NPZ arrays, and the
full checker compares targets to author files and independently recomputes errors. Reference
selection uses the standard final export, distinct from Asterion validation-force selection.
Equal parameter counts or identical reference selection criteria are not claimed.

MACE serializes its export as a full Torch module. Only the separate reference process loads its
own newly trained export. Asterion verification never loads that executable format; original
checkpoints remain JSON plus safetensors. The intended API is pinned to
[MACE v0.3.14 eval_configs](https://github.com/ACEsuit/mace/blob/v0.3.14/mace/cli/eval_configs.py).
