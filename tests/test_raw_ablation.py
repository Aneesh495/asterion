from copy import deepcopy

import numpy as np
import pytest

from asterion.artifacts.ablations import verify_ablations
from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.source import project_root, source_inputs
from asterion.contracts.errors import IntegrityError
from asterion.training.campaign import ablation_campaign


@pytest.fixture(scope="module")
def raw_ablation(tmp_path_factory):
    # Formula-generated file-format fixtures. These are not electronic-structure observations.
    root = tmp_path_factory.mktemp("raw-ablation")
    storage = root / "storage/rmd17"
    storage.mkdir(parents=True)
    distance = np.random.default_rng(82).uniform(0.8, 1.2, 10000)
    xyz = np.zeros((len(distance), 2, 3))
    xyz[:, 1, 0] = distance
    force = np.zeros_like(xyz)
    force[:, 0, 0] = 40 * (distance - 1)
    force[:, 1, 0] = -force[:, 0, 0]
    path = storage / "rmd17_ethanol.npz"
    np.savez(
        path,
        coords=xyz,
        energies=20 * (distance - 1) ** 2,
        forces=force,
        nuclear_charges=[6, 1],
        old_indices=np.arange(len(distance)),
    )
    np.savetxt(storage / "index_train_01.csv", np.arange(1000), fmt="%d", delimiter=",")
    np.savetxt(storage / "index_test_01.csv", np.arange(1000, 2000), fmt="%d", delimiter=",")
    config = read_json(project_root() / "configs/ci.json")
    config.update(budgets=[8], batch_size=4)
    ablation_campaign(config, root / "storage", root / "ablations")
    return root, config, source_inputs(), {"rmd17_ethanol.npz": path}


def verify(case):
    root, config, source, datasets = case
    return verify_ablations(root, "ablations/ablation-units.json", config, source, datasets)


def test_real_raw_ablation_replays_five_saved_model_interventions(raw_ablation):
    result = verify(raw_ablation)
    assert len(result) == 5 and all(row["probes"] == 8 for row in result)


def test_raw_ablation_summary_reduction_allows_only_numerical_roundoff(raw_ablation):
    root, _, _, _ = raw_ablation
    path = root / "ablations/ablation-units.json"
    saved = read_json(path)
    perturbed = deepcopy(saved)
    perturbed[0]["metrics"]["trained"]["force_mae_eV_angstrom"] += 1e-16
    try:
        write_json(path, perturbed)
        assert len(verify(raw_ablation)) == 5
        perturbed[0]["metrics"]["trained"]["force_mae_eV_angstrom"] += 0.01
        write_json(path, perturbed)
        with pytest.raises(IntegrityError, match="summary metric"):
            verify(raw_ablation)
    finally:
        write_json(path, saved)


def test_raw_ablation_rejects_hidden_failed_seed(raw_ablation):
    root, _, _, _ = raw_ablation
    path = root / "ablations/ablation-units.json"
    saved = read_json(path)
    try:
        write_json(path, saved[:-1])
        with pytest.raises(IntegrityError, match="population"):
            verify(raw_ablation)
    finally:
        write_json(path, saved)
