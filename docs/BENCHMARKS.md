# Benchmark methodology

`make pilot CONFIG=configs/full.json` measures the frozen architecture before research training.
The full coordinator acquires primary and secondary data, then pilots each system and both
original architectures at batch sizes one and eight. Three repetitions retain actual energy/force
and double-backward work, raw clocks, process high water and accelerator allocated-byte peaks.
Random initialized states are retained for independent output/gradient replay. No optimizer step
or supervised label access is performed. The quadratic current-output objective is a derivative
mechanism workload, not an estimate of real normalized training accuracy.

Pilot verification checks the exact source, model architecture, author coordinates, backend,
precision, sample matrix and saved-state outputs. CPU memory is not accelerator memory. A CPU
pilot using full model dimensions remains a CPU observation. No 72-hour campaign forecast is
inferred from these kernels: validation, reference retraining, preparation and I/O remain separate.

Current `make bench` loads the selected original checkpoint at the configured training precision
and device. Runtime format v2 retains all 740 samples: 30 repetitions of six modes at batch sizes
1, 2, 4 and 8, ten 100-call inference repetitions and ten 100-step learned-force dynamics starts.
It records the executing source, checkpoint, parameter fingerprint, author geometry identities,
monotonic clock intervals and exact completed work. Existing output identities are never overwritten.

The six short modes are fresh graph construction, neighbor reuse, energy plus force, force-loss
backward, a Hessian-vector product and host/device coordinate round trips. CPU transfer measurements
are copies, not accelerator transfer estimates. Short timings exclude collation. The first sample
in each inference/backward/curvature cell retains outputs for independent saved-weight replay.
Every graph sample is checked against NumPy all-pairs edge counts. Repeated coordinates reuse the
cache after the first rebuild at each batch size.

Dynamics timings include collation, neighbor updates and trajectory serialization. Each run starts
from retained author coordinates with a reproducible 100 K velocity initialization and requests
100 NVE steps of 0.1 fs. Failed starts remain in the population. Throughput uses completed steps,
and the report retains failure reasons and completed physical horizons. Independent checking replays
SI integration, initial velocities, cache rebuilds and saved forces along each trajectory. These
observations do not establish long-horizon molecular stability or matched reference performance.

Full verification requires the complete matrix, declared research backend and precision, retained
evaluation geometries, genuine model outputs and raw trajectories. It rejects shortened work,
duplicate cells, changed durations, invented memory, substituted forces/curvature and inflated
dynamics throughput. The current CUDA measurement path remains unexecuted on this host.

## Historical CPU timing

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

Full research measurements still need a device memory/resource pilot, long-horizon dynamics,
matched reference physical horizons and accuracy, accelerator transfer costs, acquisition total cost and
matched reference accuracy. Energy-only inference cannot be compared to energy-plus-force costs.
No 72-hour full campaign completion estimate has been measured.

Acquisition cost reports are separate from microbenchmarks. They count all three trained members
per round and retain actual optimizer/example totals, epoch durations and call clock intervals.
Candidate inference and calibration/evaluation timing are included in round observations.
Shared preparation and final summary writes are declared exclusions. Read-only checking binds
these records to histories and author-coordinate diversity. Actual paired policy outcomes are
recomputed from held-out errors; timing does not imply better label efficiency.
