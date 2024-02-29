# Reproducibility and commands

```sh
make bootstrap
make doctor
make build
make test
make data
make train RUN_ID=research-diagnostic
make evaluate RUN_ID=research-diagnostic
make simulate RUN_ID=research-diagnostic
make demo RUN_ID=research-diagnostic
make diagnostic RUN_ID=research-diagnostic EVIDENCE=evidence/research-diagnostic
make verify RUN_ID=research-diagnostic EVIDENCE=evidence/research-diagnostic
make controls RUN_ID=research-diagnostic EVIDENCE=evidence/research-diagnostic
```

Executable inputs must be committed before diagnostic evidence generation. A new evidence path
is required for each generation. Source changes invalidate a prior bundle; verification never
updates hashes or repairs observations. Raw data remain in ignored storage and are required for
independent unit and identity checks. Reacquire using the pinned author release manifest.

`make verify EVIDENCE=... DATA=/path/to/storage` resolves author files beneath that storage
directory without editing the bundle's recorded paths or hashes. Primary files must be at
`rmd17/<pinned filename>` and MD22 at `md22/<pinned filename>`. The verifier still checks the
committed release metadata, primary MD5 and SHA256, and secondary SHA256 and exact byte size.
An explicit storage override does not fall back to recorded paths when a file is missing.
The same override applies to controls, full verification, reference inputs and resource pilots.

`make train RUN_ID=... CONFIG=configs/full.json` selects the full original training matrix.
`make acceptance RUN_ID=...` performs full preflight and records any blocked gate without
claiming success. The full stage coordinator and candidate promotion are implemented but the complete campaign
remains unverified. An accelerator is required by the full resource profile. No paid compute is provisioned by these commands.

The package pins Python 3.12.14, PyTorch 2.8.0 and e3nn 0.5.7. The uv lock pins transitive versions
and artifact hashes. CPU diagnostic arithmetic is float64. Deterministic CPU continuation is
checked through exact tensor comparisons. Accelerator execution has a numerical contract and
must not be represented as bitwise deterministic without measurement.

Independent commands include `make properties`, `make derivatives`, `make analytic`, `make bench`,
`make stress`, `make acquire`, `make ablate`, `make secondary`, and `make reference`. Expensive
commands retain raw outputs; they do not certify the complete campaign by themselves.

Run `make acquire RUN_ID=...` before `make demo RUN_ID=...` to attach independently checked
acquisition rounds to the local viewer. Full acceptance runs these stages in that order.
See [viewer observations](VIEWER.md) for displayed quantities, costs and unavailable states.
