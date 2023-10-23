# Recovery runbook

Each training checkpoint is an immutable directory published by atomic rename. Tensors and JSON
are written before publication. The JSON identity includes tensor hashes. The latest pointer is
written only after the generation exists. A run coordinator owns the directory. Do not run two
training processes against one identity.

Resume with `make train RUN_ID=...` plus the CLI `--resume` option, or invoke the library with an
explicit checkpoint path. Resume verifies architecture, data, split, reveal and source lineage.
The CPU sampler is a deterministic permutation of seed and epoch. Optimizer moments and Torch
RNG state are restored. Changing the frozen workload is a new run, not a resume.

Every new generation embeds the validation history and selection through its own epoch.
Selection is published before the latest pointer. If a crash publishes a generation without
advancing latest, recovery moves future generations and mutable summaries into `recovery/`.
It restores history, selection and latest from the immutable resume generation. `RECOVERY.json`
records intent before moving files, so a second interruption can finish the archive transaction.
Retained bytes are checked before each move. Partial initialization without a generation is moved
to `initialization-attempts/` before starting again. Completed summaries cannot bypass full lineage
checking. Checkpoint loading preserves the caller's Torch RNG state.

The full coordinator converts SIGINT and SIGTERM into cancellation, terminates its own stage
process group and retains the actual exit, logs and interruption state. Resume verifies the stage
definition. Passed stages require unchanged output hashes. Training resumes committed optimizer
work. Stages without a continuation mechanism archive owned partial outputs and restart from
their declared inputs; failed attempts remain in `STAGES.json` and `stage-attempts/`.
After SIGKILL, a live orphan child is never duplicated. Let that owned process finish before resume.
CUDA checkpoints preserve RNG state but do not promise bitwise continuation on a different device
population. CUDA recovery has not been executed on this host.

Simulation checkpoints retain all frames through the saved step, configuration, model identity,
array hash and NumPy generator state. A crash between array publication and the resume pointer
leaves an unused immutable generation; the last valid pointer remains authoritative. NVT noise
draws resume exactly on CPU. Neighbors are rebuilt after evaluator recreation; their physics must
match even if rebuild counters differ for a restarted learned evaluator.

Failed experiments retain their files and error. Choose a new run identity for repaired source.
Run `make stress` for analytic simulation recovery, force-loss training recovery, invalid geometry
rejection and 30-minute inference memory sampling. Raw measurements remain outside Git.
