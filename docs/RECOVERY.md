# Recovery runbook

Each training checkpoint is an immutable directory published by atomic rename. Tensors and JSON
are written before publication. The JSON identity includes tensor hashes. The latest pointer is
written only after the generation exists. A run coordinator owns the directory. Do not run two
training processes against one identity.

Resume with `make train RUN_ID=...` plus the CLI `--resume` option, or invoke the library with an
explicit checkpoint path. Resume verifies architecture, data, split, reveal and source lineage.
The CPU sampler is a deterministic permutation of seed and epoch. Optimizer moments and Torch
RNG state are restored. Changing the frozen workload is a new run, not a resume.

Simulation checkpoints retain all frames through the saved step, configuration, model identity,
array hash and NumPy generator state. A crash between array publication and the resume pointer
leaves an unused immutable generation; the last valid pointer remains authoritative. NVT noise
draws resume exactly on CPU. Neighbors are rebuilt after evaluator recreation; their physics must
match even if rebuild counters differ for a restarted learned evaluator.

Failed experiments retain their files and error. Choose a new run identity for repaired source.
Run `make stress` for analytic simulation recovery, force-loss training recovery, invalid geometry
rejection and 30-minute inference memory sampling. Raw measurements remain outside Git.
