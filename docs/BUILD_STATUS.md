# Build status

Phase: source-bound CPU diagnostic verified; full specification incomplete.

Execution commit is `6e95f9544bf733981a4cb192acec6475cadb15cb`. Executable input digest is
`e94aeb4d5201d65c86d6514305f7b796d5fc2ce7bbae1eeff10f06b328433283`.
Later documentation/package commits do not replace this training source identity.
No GitHub remote or hosted publication exists. `gh auth status` found the Aneesh495 keyring token
invalid. The host is an Apple M2 MacBook Air with 16 GiB shared RAM and no CUDA.

## Executed evidence

| Observation | Status | Retained identity |
| --- | --- | --- |
| Locked installation, Ruff formatting/lint and mypy | passed | Source 6e95f95, Python 3.12.14, Torch 2.8.0, e3nn 0.5.7 |
| Mathematical/semantic/integration tests | passed | 164 collected cases; `runs/validation-02/tests.xml` |
| Generated graph properties | passed | 10,000 cases, `runs/validation-02/properties.json` |
| Original force training | passed | `frozen-diagnostic-02`: six runs, three seeds, 100 labels, 100 epochs, 1300 updates each |
| Held-out physical metrics and saved weights | passed | 256 geometries/run, verifier replays 48 held-out model probes |
| Symmetry and derivative campaign | passed | 1200 symmetry, 1000 force, 100 Hessian checks, `runs/frozen-diagnostic-02/derivatives/` |
| Analytic dynamics at matched horizons | passed | 110 fixtures, three timesteps, `runs/validation-02/analytic/` |
| Learned dynamics | passed | Nine 250-step starts, 2250 completed steps, 25 fs/start |
| Actual fail-closed demo | passed | `runs/frozen-diagnostic-02/demo-failure/`, reason `extreme-force` |
| Acquisition integration | passed | `acquisition-integration-02`: three policies, 8/16/32 labels, 27 two-epoch members |
| Architectural ablation integration | passed | `ablation-integration-02`: five two-epoch variants |
| MD22 separate-model adapter integration | passed | `secondary-integration-02`: 32 labels, 64 held-out geometries, two 16-step starts |
| CPU interruption/recovery | passed | 50 simulation plus five training cases, `runs/frozen-diagnostic-02/*-recovery/` |
| Invalid/resource/failure cases | passed | 100 actual rejections, `runs/frozen-diagnostic-02/faults/` |
| Raw benchmark | passed | 490 samples, `runs/frozen-diagnostic-02/benchmark.json` |
| Diagnostic bundle verification and controls | passed | `evidence/frozen-diagnostic-02`, 16 negative exits, 13 rehashed; valid before/after |
| Documentation and diagrams | passed | Eight ADRs, ten editable/rendered DOT diagrams, four source-linked walks, raw figures |
| Two clean installations/builds | passed | `runs/validation-02/clean-builds.json`, matching wheel and sdist for execution source |
| 30-minute inference stress | unverified | Running to its declared horizon; only final `memory-stress.json` can establish completion |

The equivariant diagnostic's force MAE was 0.111071, 0.131622 and 0.129603 eV/angstrom,
an 84.9 to 87.1 percent reduction versus zero force. Tiny acquisition/ablation/MD22 integration
runs did not demonstrate meaningful learning or policy advantage. Every outcome is retained.
See [findings](FINDINGS.md), [claim map](CLAIMS.md) and [capability register](CAPABILITIES.md).

## Full gates

| Gate | Status | Outstanding work |
| --- | --- | --- |
| MA01 | failed | Only 33 authored test functions; 164 parameterized cases do not satisfy 400 authored tests |
| MA02 | unverified | All four primary files/splits acquired, but full budget/reveal campaign absent |
| MA03 | unverified | Float64 diagnostic count met; full source/coverage assembly absent |
| MA04 | unverified | Diagnostic derivative counts met; full coverage/precision campaign absent |
| MA05 | blocked | 72 original full runs and 12 pinned MACE retrainings require unavailable accelerator; reference environment unverified |
| MA06 | unverified | Full 5000-geometry protocols, learning confidence intervals and raw shift checks absent |
| MA07 | unverified | Diagnostic analytic population met; complete full bundle absent |
| MA08 | blocked | One million valid steps and 90% 10,000-step-start completion absent |
| MA09 | blocked | Required two-molecule, three-seed, 100/300/1000-label acquisition campaign absent |
| MA10 | blocked | Full ablation/uncertainty study and 10 by 5000-step MD22 study absent |
| MA11 | unverified | Recovery/fault counts met locally; final memory run and full matched scaling pending |
| MA12 | unverified | Full positive assembly, controls, release and hosted source verification absent |

The implementation-depth requirement is also unmet. No padding or lowered gate is used.
`evidence/full-preflight-02/ACCEPTANCE.json` records a blocked full profile and a nonzero command
exit. Full acceptance assembly is unfinished even on suitable hardware. A blocked report is not
a passed artifact. No paid compute was provisioned or requested as part of this run.

## Continuation

Resume from this source, not a new skeleton. First finish connected irrep feature tracing,
full stage orchestration/assembly, raw reference/secondary/shift validation, paired block learning
intervals, acquisition viewer lineage and substantive independent tests. The exact full population
is frozen in `configs/full.json`; do not lower it to fit this host. Commit fixes, create a new run,
and rerun source-bound execution after executable changes.

On compatible local hardware, measure the resource pilot and separate MACE environment, then
execute full training, evaluation, simulation, acquisition, ablations, reference, secondary,
properties, derivatives, analytic, stress and bench stages. Assemble a complete full bundle,
verify it, execute semantic controls, and package only after those gates pass.

The completed diagnostic can be checked now with:

```sh
make verify RUN_ID=frozen-diagnostic-02 EVIDENCE=evidence/frozen-diagnostic-02
make controls RUN_ID=frozen-diagnostic-02 EVIDENCE=evidence/frozen-diagnostic-02
```

Raw local files remain ignored, accessible and hash-bound. Old source-mismatched pilots are retained
as historical diagnostics and are not used to certify current execution.
