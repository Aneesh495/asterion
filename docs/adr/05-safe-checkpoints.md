# Publish tensor generations atomically

Status: implemented. See [recovery protocol](../RECOVERY.md).

A completed filename must identify one consistent model, optimizer, RNG state and scientific
lineage. Training writes a new generation into an owned temporary directory, flushes its files,
and atomically renames the directory before updating the latest pointer. Immutable generations
remain inspectable after interruption. Resume verifies source, configuration, author data, split,
reveal chain and saved training progress before any next update.

Pickle-based state dictionaries are widely supported but can execute code when loaded. The chosen
format stores model and optimizer tensors with safetensors and uses a bounded JSON container tree
for non-tensor structure. A weights-only export would reduce storage but cannot reproduce optimizer
momentum, sample ordering or stochastic continuation. Overwriting one checkpoint in place saves
filenames while exposing partially written state and destroying the prior recovery point.

Separate generations cost disk space, metadata operations and write bandwidth. They also make
retained optimizer and RNG state auditable. CPU recovery compares uninterrupted and interrupted
updates exactly. CUDA can use nondeterministic kernels, so that CPU result does not imply bitwise
accelerator continuation. The full accelerator run must retain the declared numerical contract.

A crash before directory publication leaves an incomplete temporary generation rather than a
completed checkpoint. A crash after publication but before pointer update leaves a complete older
pointer and a recoverable immutable generation. Corrupt tensor hashes, invalid tree references or
lineage mismatches fail loading. The campaign and evidence publication use single-owner locks,
retain partial attempts, and refuse to replace an unrelated live owner or an existing final output.
