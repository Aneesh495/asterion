# Build status

Phase: original scientific implementation and diagnostic validation.

- Passed: read the requested specification; confirmed a fresh workspace.
- Passed: verified author rMD17 release v4, DOI 10.6084/m9.figshare.12672038.v4, CC0.
- Blocked: GitHub authentication. `gh auth status` reports an invalid Aneesh495 keyring token.
- Blocked: full accelerator profile. Available host is an Apple M2 with 16 GiB shared memory.
- Passed: original invariant and O(3) equivariant scalar-energy models, energy-gradient forces,
  force-loss double-backward, radius graphs/cache, physical units, split/reveal interfaces.
- Passed: 151 collected tests covering numerical differences, symmetry, dynamics and label access.
- Passed: three-seed ethanol pilot for both original models at 100 labels and 100 epochs.
  Equivariant held-out force error fell by 87.6 to 88.5 percent versus zero force on 256 geometries.
  These pilot runs preceded final source freezing and cannot certify full acceptance.
- Implemented, unverified end-to-end: acquisition, ablations, secondary/reference adapters, verifier,
  viewer, benchmark/property/resource campaigns and source-bound diagnostic assembly.
- Unverified: fixed full research campaign and publication. No complete acceptance claim is made.

The requested spec's workload thresholds remain requirements. Diagnostic execution cannot
certify the full campaign. No paid compute has been provisioned. Public publication has not occurred.

Next action: complete type/integration checks, freeze source, retrain the diagnostic profile,
run retained dynamics, then execute positive and negative verifier controls.
