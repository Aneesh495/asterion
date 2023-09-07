# Diagnostic model card

The retained models are original randomly initialized energy potentials trained on rMD17 ethanol,
release v4, with the PBE/def2-SVP label level. They support the released neutral singlet H/C/N/O
nonperiodic scope. The measured validity domain is ethanol near the sampled geometries. Accepting
an element does not establish accuracy on another molecule, reaction, charge state or spin state.

Run `frozen-diagnostic-02` contains three equivariant and three invariant models. Seeds are 1, 2
and 3. Each receives exactly 100 distinct energy/force label pairs. Every run uses 100 epochs,
batch size 8, Adam with cosine-decayed learning rate starting at 0.003, normalized energy weight
0.1 and force weight 1.0, CPU float64, width 8, depth 2, eight radial channels and a 5 angstrom
cutoff. The equivariant model includes 0e, 1o and 2e channels and a quadratic invariant readout.

The offset and scales are fit only on revealed training labels. Minimum validation force MAE
selects the checkpoint. Validation and calibration each contain 64 distinct geometries. Primary
evaluation contains 256 per seed, and a separately named temporal shift contains 251 to 255.
The author split is correlated. The shift experiment excludes fitting and primary evaluation
rows, but its final count is reduced by exclusions. This is not the requested full shift workload.

Selected checkpoints live under `runs/frozen-diagnostic-02/<model>-ethanol-100-<seed>/`, where
`selection.json` identifies an immutable `epoch-NNNNN` directory. `checkpoint.json` records tensor
hashes and origin; weights are safetensors, never executable pickle. `initial/` retains random
weights. `history.json`, `split.json`, the reveal chain and raw predictions establish provenance.

Execution source is `6e95f9544bf733981a4cb192acec6475cadb15cb`, digest
`e94aeb4d5201d65c86d6514305f7b796d5fc2ce7bbae1eeff10f06b328433283`.
Raw files are local ignored artifacts, not weights downloaded from another potential.
The diagnostic verifier checks dataset units, selection, reveal count, nonzero force-loss
gradients, parameter changes and eight source-selected saved-weight predictions per run.

[Findings](FINDINGS.md) give every seed's errors. The 25 fs trajectories demonstrate integration
and short-horizon detector behavior only. No diffusion, equilibrium free energy, reaction barrier,
thermodynamic accuracy or long-horizon stability result has been established. A locally conservative
force can still be inaccurate. The single-model demo has unavailable committee uncertainty and
says so. An uncertainty threshold with unavailable uncertainty terminates explicitly.

Long-range interactions, periodic boundaries, variable charge/spin, elemental energy transfer,
accelerator precision and cross-platform numerical behavior remain outside validated capability.
The reference MACE and secondary MD22 campaigns are unexecuted. Original code is MIT licensed.
The upstream rMD17 label license is recorded separately in the [data card](DATA.md).
