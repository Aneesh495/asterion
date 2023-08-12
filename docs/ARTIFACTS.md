# Artifact schema and verification

Training runs retain configuration, source inputs, immutable split, reveal chain, training-only
normalization, random initial weights, optimizer state, epoch histories, selected checkpoints,
completed counts and raw held-out predictions. Tensor weights use safetensors. JSON tree nodes
describe optimizer containers and refer to separately stored tensors. Loading never uses pickle.

A diagnostic bundle has SOURCE_INPUTS.json, DATASETS.json, ENVIRONMENT.json, CAMPAIGN.json,
COMMANDS.jsonl, UNITS.json, MANIFEST.json and ACCEPTANCE.json. Every included artifact has a SHA256.
The verifier rejects extra or missing files. It compares executable bytes with the source commit
and current checkout, then validates model/data/split/reveal lineage and selection integrity.

Metrics are recomputed with independently expressed arithmetic. Labels and positions are checked
against author data with independently derived unit conversion. At least eight probes per run
are chosen by source hash and geometry identity, then replayed with actual saved weights. Initial
and selected weights must differ. The verifier never trains, regenerates missing observations,
changes hashes or repairs a bundle.

Trajectory artifacts retain every completed frame, noise used by NVT, velocities, forces,
potential and kinetic energy, temperature, heat, uncertainty, neighbor rebuilds and termination.
A failed candidate is retained separately. Independent SI reconstruction checks each completed
integration and reports drift. An explicit failed trajectory can be a valid detector observation.

Negative controls operate on disposable copies. Semantic controls update hashes before invoking
the verifier. Their nonzero exits and diagnostics are retained. Diagnostic controls validate the
diagnostic verifier, not the still-pending full research evidence verifier.
