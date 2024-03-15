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

An installed wheel locates the calling source checkout through the named pyproject marker.
Before recording source lineage, its installed Python/HTML input inventory must match the
checkout's src inventory exactly. A stale wheel cannot record newer source hashes. Direct model
inference can use the package outside a checkout; repository/evidence commands require a checkout.

Trajectory artifacts retain every completed frame, noise used by NVT, velocities, forces,
potential and kinetic energy, temperature, heat, uncertainty, neighbor rebuilds and termination.
A failed candidate is retained separately. Independent SI reconstruction checks each completed
integration and reports drift. An explicit failed trajectory can be a valid detector observation.

An invalid initial output has no valid physical frame. Format
`asterion-rejected-initial-v1` stores raw coordinates, velocities and elements in `failure.npz`.
Returned force/energy/uncertainty values are retained even when nonfinite or malformed; an
evaluation exception instead retains its type/message with outputs unavailable. Metadata records
zero completed steps, zero valid frames and zero physical horizon. No `trajectory.npz` is created.
Replay checks the raw reason and source inputs; saved-model checking reproduces the output or
exception. Energy drift is unavailable for this empty physical trajectory. The viewer shows
initial coordinates with unavailable force/energy and empty histories.

Negative controls operate on disposable copies. Semantic controls update hashes before invoking
the verifier. Their nonzero exits and diagnostics are retained. Diagnostic controls validate the
diagnostic verifier, not the still-pending full research evidence verifier.

Full evidence has candidate and final phases. A candidate contains every scientific raw family
and has candidate status. Read-only candidate verification enforces full thresholds while requiring
controls to be absent. Controls invoke that same checker before, during and after corruption.
Promotion creates a new final directory only when candidate and control manifest identities match.
Default full verification and release reject candidate status. The candidate remains unchanged.

Derivative campaigns retain original coordinates/elements, orthogonal transforms, permutations,
translations, system energies/forces, central-difference energies, directional forces and Hessian
products. Their index binds the actual saved checkpoint, executed float64 state fingerprint and
source. The full checker independently reconstructs force differences, curvature, covariance,
force/torque, additivity and batching errors from these values. It validates author coordinates
and seeded perturbations, then replays eight source-selected cases per family using saved weights.
It does not invoke the producer's numerical-check routines or regenerate missing traces. JSON
chemistry element lists are restored to immutable tuples at the checker boundary.

Full bundles retain `STAGES.json`, original stage logs and `STAGE_OUTPUTS.json`. The latter maps
execution paths to copied observations without changing their bytes. Training receipts include
all immutable generations, initialization, split/reveal/normalization and completed history.
Evaluation receipts include predictions, baselines, intervals and shift observations. Other
scientific stages retain their whole observation directories. A completed index alone cannot
hide changed weights or absent raw files. Verification checks actual exit status, log hashes,
elapsed budgets, exact observation ownership, stage order and successful continuation after
retained failures. The inner CLI command must agree with its managed process definition.

Full evidence also retains `FEATURE_COVERAGE.json`. The committed
`configs/features.json` maps M01 through M32 to source files, authored tests, actual commands,
raw families and MA gates. Verification discovers module test functions and class methods from
the frozen source, checks every one against retained JUnit cases, rejects skipped or repeated
cases, and reconstructs the complete map. Every production file and authored test file must be
mapped. Artifact inventories and successful managed command definitions bind the observations.
`source-test-artifact-bound` describes structural coverage; independent scientific gate checks
still decide acceptance. A mapping cannot make missing full execution pass.

The documentation stage validates links, public prose and existing diagram labels without editing
source files. Its retained file hashes must match the frozen documentation. `make docs` explicitly
regenerates diagram exports before a source freeze. Source identities now include documentation,
README, license and CI definitions, so a changed mathematical contract invalidates prior evidence.

`make release EVIDENCE=... RUN_ID=...` requires a full final bundle that passes independent
verification. It builds the wheel from the pinned source distribution, retains build receipts,
and packages the full raw evidence, safe checkpoint generations, selected-model index, source
snapshot, Git source bundle, dependency locks and reproduction instructions. Author arrays stay
external with pinned hashes and explicit license states. The archive normalizes member metadata
and reads every payload back before committing a new filename. It refuses existing outputs,
symlinks, path traversal, duplicate members, missing bytes and insufficient disk space. Failed
attempts and logs remain in owned staging directories. Archive-integrity verification checks
container bytes only; scientific verification must still run against external author data.
The full release path remains unexecuted while full CUDA acceptance is blocked.
