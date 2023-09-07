# Data card and splitting

rMD17 release v4 is pinned by the author metadata in `configs/rmd17-release-v4.json`. The required
individual files are aspirin, ethanol, malonaldehyde and salicylic. Each has 100,000 geometries.
The released name for salicylic acid is `rmd17_salicylic.npz`. Author MD5 and local SHA256 identities
are checked before use. The release license is CC0. Its theory level is PBE/def2-SVP.

[Author release](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
describes time-series correlation and limits meaningful training to at most 1000 structures.
Budgets count unique structures. Rotations, force components, repeated epochs and retries do not
increase the label budget.

Author CSVs contain 1000 training and 1000 test rows. The author-plus-evaluation-supplement protocol
preserves the published membership and extends evaluation with separately identified remaining
rows when more than 1000 are requested. Validation and calibration are additional disjoint rows.
This protocol has no claim of temporal independence. The nearest training descriptor distances
and original-index gaps are retained to expose correlated evaluation.

`guarded_split` creates ordered temporal blocks with explicit guards. Species-pair sorted distance
descriptors audit near duplicates without using labels. These descriptors are invariant to rigid
motion and same-species permutations but are not a complete conformational representation. Geometry
near-duplicate counts are reported; they are not silently excluded to improve a metric.

MD22 Ac-Ala3-NHMe contains 85,109 geometries and 42 atoms. Its author file declares PBE+MBD at 500 K.
The SHA256 in `configs/md22-release.json` defines the acquired file version. A separate model and
ledger are required because its label level differs from rMD17. License terms for redistributing
this file remain unverified, so data are not shipped with the package. The bounded adapter checks
file bytes and converts the declared units. A 32-label, two-epoch adapter integration run completed
64 held-out evaluations and two 16-step starts. It is not the required extended-system study,
which remains unexecuted.

[MD22 author download](https://sgdml.org/) and the
[extended-molecule study](https://www.nature.com/articles/s41467-023-39214-w) provide provenance.
