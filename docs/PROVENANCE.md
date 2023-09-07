# Licenses and provenance

Original Asterion code is MIT licensed in the root LICENSE. PyTorch, NumPy, SciPy, e3nn,
safetensors and other dependencies retain their upstream licenses. They are locked dependencies,
not included as authored production code. e3nn provides tensor products, irreps and spherical
harmonics; the model interaction, readout, data, training, acquisition and dynamics code is original.

rMD17 author release v4 is pinned in `configs/rmd17-release-v4.json`, including individual file IDs,
byte sizes, MD5 checksums and release metadata. Local `storage/rmd17/DATASETS.json` adds SHA256 and
acquisition paths. All four required molecular files and seeds 1 to 3 of the author CSV splits were
acquired and verified. Original indices remain in the coordinate pool. The author metadata declares
CC0. Dataset arrays are not included in wheel or source archives.

[rMD17 author release](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
and [author metadata](https://api.figshare.com/v2/articles/12672038) supply the pinned provenance.
The release's kcal/mol energy/force units are converted independently during verification.

MD22 Ac-Ala3-NHMe was acquired from the author site with SHA256
`c9332a2f8d3868cd4d5f259e8e1c90497b4a0b17705c00aae1f86bf844c9f6b2`.
Its file declares PBE+MBD and 500 K. `configs/md22-release.json` records its URL, bytes and hash.
Its redistribution license remains unverified, and neither data nor models trained on it are
included in a release. [Author site](https://sgdml.org/) and
[MD22 study](https://www.nature.com/articles/s41467-023-39214-w) are references.

MACE v0.3.14 is a separate prospective reference, with an external environment and subprocess
adapter. It supplies no production Asterion energy or forces. No pretrained checkpoint, publication
accuracy or imported training result is presented as an original observation. Its retraining is
unexecuted. [MACE releases](https://github.com/ACEsuit/mace/releases) identify the intended version.

Diagnostic plots derive from local original training and integration outputs. CSV tables and
`docs/figures/PROVENANCE.json` retain input hashes, run, execution commit and selected checkpoint
identities. The viewer embeds actual coordinates, predicted/reference forces and retained energy
histories. No publication figure or generated molecular image is used as measured evidence.
