# Benchmark methodology

`make bench RUN_ID=frozen-diagnostic-02` loads the selected original seed-1 equivariant checkpoint
and retained ethanol geometries. It records raw timings, precision, device, atom count, graph edge
count where applicable, repetition and cumulative native process peak RSS. The measured model is
width 8, depth 2, lmax 2, radial 8 and float64 on CPU. Torch uses two threads.

Four modes run 30 repetitions for batch sizes 1, 2, 4 and 8 geometries, each containing nine atoms.
`graph` builds fresh topology and differentiable vectors. `cached_graph` reuses a neighbor cache
on the same coordinates and includes its validity check; its first call can rebuild. `energy_force`
includes graph construction, scalar inference and coordinate gradient. `force_loss_backward` also
differentiates a force loss through model parameters. It measures the derivative mechanism on
retained coordinates, not a timed optimizer update on labeled training batches.

Every short sample constructs its batch before the timed section. GPU synchronization is implemented
but unexecuted. Ten long repetitions each perform 100 energy-plus-force calls; those measurements
include collation. No short sample is silently dropped. These local diagnostics ran alongside
other tasks and are not isolated machine throughput estimates.

| Batch geometries | Fresh graph ms | Cached graph ms | Energy plus force ms | Force-loss backward ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.207 | 0.145 | 19.515 | 36.060 |
| 2 | 0.288 | 0.182 | 16.791 | 43.066 |
| 4 | 0.444 | 0.200 | 22.419 | 63.787 |
| 8 | 0.741 | 0.248 | 35.670 | 95.527 |

Table entries are medians of 30 raw samples per cell, not confidence intervals. Ten 100-call
repetitions had median elapsed time 1.517 seconds. `runs/frozen-diagnostic-02/benchmark.json`
contains all 490 samples, including outliers and exact completed-call counts. The batch-1
energy-plus-force maximum was 315.777 ms, so the median alone hides substantial local variability.

CPU native peak RSS is bytes on macOS and KiB on Linux. It is cumulative process high water,
not current tensor live bytes. The 30-minute `make stress` measures repeated energy/force inference
on a retained genuine geometry and samples peak RSS every five seconds. Its final report records
actual elapsed time and completed calls. Missing final output is an incomplete stress run.

Full research measurements still need a device memory/resource pilot, true dynamics throughput,
matched physical horizons, transfer costs, cache behavior during motion, acquisition total cost and
matched reference accuracy. Energy-only inference cannot be compared to energy-plus-force costs.
No 72-hour full campaign completion estimate has been measured.
