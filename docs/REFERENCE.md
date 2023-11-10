# Independent reference comparison

MACE is a separately labeled reference implementation, pinned to mace-torch 0.3.14. It is not
imported into Asterion models or production accounting. The reference adapter exports the same
revealed structures as extended XYZ in physical units, fixes validation rows and retains exact
command arguments, logs, file hashes and process exits.

Create the comparison environment explicitly:

```sh
uv venv --python 3.12 .reference-venv
uv pip sync --python .reference-venv/bin/python --require-hashes requirements/reference-linux.txt
make reference CONFIG=configs/full.json RUN_ID=full-research
```

This command requires Linux CUDA and uses 1000 training structures per molecule and seed.
For the macOS CPU compatibility environment, use `requirements/reference-darwin.txt` instead.
The probe checks all resolved distribution versions, Python, CUDA and reference CLI source bytes.
The macOS lock was installed and checked. A separate CPU run trained MACE's declared 24-channel,
two-interaction, correlation-three model for two epochs on 16 ethanol geometries, exported it
and evaluated eight geometries. The two actual commands exited zero; the raw training journal
contains eight optimizer updates and 32 processed examples. Its held-out force MAE was
0.805081 eV/angstrom. This checks compatibility and is not the full matched comparison.
The first attempt failed because its export directory was absent; that attempt is retained.
The adapter now creates its output directories. The Linux CUDA environment and complete
400-epoch retrainings remain unexecuted. Their compatibility, checkpoints and prediction
reproduction must be established before MA05 can pass. No pretrained
scores from the MACE publication are used as observed Asterion results.

Labels use REF_energy and REF_forces to avoid ASE reserved-key relocation. After training,
the named standard final export is evaluated with the pinned eval_configs CLI. Retained XYZ
has PRED fields and original identities. A bounded original parser saves NPZ arrays, and the
full checker compares targets to author files and independently recomputes errors. Reference
selection uses the standard final export, distinct from Asterion validation-force selection.
Equal parameter counts or identical reference selection criteria are not claimed.

Raw MACE result journals retain every optimizer batch and each validation epoch. The checker
requires 400 epochs, 125 actual batches per epoch and 1000 unique author training rows. Exported
training, validation and evaluation XYZ labels are checked against the pinned author arrays.
Each input role retains its geometry identities and physical units. Rounded ASE evaluation
exports use a separately declared 1e-8 comparison tolerance; authored input XYZ remains full
precision. A metadata version string alone cannot certify executed training.

MACE serializes its export as a full Torch module. Only the separate reference process loads its
own newly trained export. Asterion verification never loads that executable format; original
checkpoints remain JSON plus safetensors. The intended API is pinned to
[MACE v0.3.14 eval_configs](https://github.com/ACEsuit/mace/blob/v0.3.14/mace/cli/eval_configs.py).
