from types import SimpleNamespace

import numpy as np
import pytest

from asterion.acquisition.analysis import (
    interval,
    paired_policy_outcomes,
    round_cost,
    selected_diversity,
)
from asterion.artifacts.acquisition_analysis import (
    verify_cost,
    verify_diversity,
    verify_interval,
    verify_paired_policies,
)
from asterion.artifacts.io import atomic_npz, read_json, write_json
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool
from asterion.data.splits import geometry_descriptors


def test_clock_interval_rejects_unobserved_or_reverse_work():
    with pytest.raises(ContractError, match="increasing"):
        interval(20, 10)


def test_clock_verifier_rejects_duration_edited_without_raw_clock():
    record = interval(1, 1000000001)
    record["seconds"] = 2
    with pytest.raises(IntegrityError, match="clock readings"):
        verify_interval(record)


def test_clock_verifier_rejects_phase_outside_its_round():
    with pytest.raises(IntegrityError, match="escapes"):
        verify_interval(interval(5, 1000000005), interval(10, 1000000010))


@pytest.fixture
def measured_cost(tmp_path):
    members = [f"round-0-member-{member}" for member in (1, 2, 3)]
    for member in members:
        write_json(
            tmp_path / member / "history.json",
            [
                {"elapsed_seconds": 0.1, "updates": 2, "examples_processed": 8},
                {"elapsed_seconds": 0.2, "updates": 4, "examples_processed": 16},
            ],
        )
    calls = [interval(1000000000 * i + 1, 1000000000 * (i + 1) + 1) for i in range(3)]
    cost = round_cost(
        tmp_path,
        0,
        interval(1, 5000000001),
        calls,
        None,
        interval(3000000001, 4000000001),
        interval(4000000001, 4500000001),
    )
    return tmp_path, {"round": 0, "members": members, "cost": cost}


def test_cost_checker_reconstructs_retained_member_counts(measured_cost):
    root, row = measured_cost
    verify_cost(root, row)
    assert row["cost"]["completed_optimizer_updates"] == 12
    assert row["cost"]["processed_training_examples"] == 48


def test_cost_checker_rejects_inflated_updates_with_coherent_summary_file(measured_cost):
    root, row = measured_cost
    row["cost"]["completed_optimizer_updates"] += 1
    write_json(root / "cost-0.json", row["cost"])
    with pytest.raises(IntegrityError, match="total cost"):
        verify_cost(root, row)


def test_cost_checker_rejects_removed_member_work(measured_cost):
    root, row = measured_cost
    row["cost"]["members"].pop()
    write_json(root / "cost-0.json", row["cost"])
    with pytest.raises(IntegrityError, match="committee member"):
        verify_cost(root, row)


def test_diversity_recomputes_permitted_coordinate_descriptors():
    coordinates = np.zeros((8, 2, 3))
    coordinates[:, 1, 0] = np.linspace(0.8, 1.2, 8)
    pool = CoordinatePool(
        np.array([6, 1]), coordinates, np.arange(8), "fixture", "bond", Chemistry()
    )
    descriptors = geometry_descriptors(pool, tuple(range(8)))
    result = selected_diversity(descriptors, range(8), [0, 2, 7])
    verify_diversity(
        result,
        coordinates,
        pool.numbers,
        list(range(8)),
        ["bond:fixture:0", "bond:fixture:2", "bond:fixture:7"],
    )
    assert result["mean_pairwise_descriptor_distance"] > result["mean_nearest_descriptor_distance"]


def test_diversity_rejects_repeated_selected_identity():
    with pytest.raises(ContractError, match="unique allowed"):
        selected_diversity(np.ones((3, 2)), range(3), [0, 0])


@pytest.fixture
def policy_arrays(tmp_path):
    pool = SimpleNamespace(original_indices=np.array([0, 1, 1000, 1001]))
    outcomes = []
    for policy, error in (
        ("uniform", [1.0, 1.0, 2.0, 2.0]),
        ("diversity", [0.8, 0.8, 1.7, 1.7]),
        ("uncertainty-diversity", [1.2, 1.2, 1.8, 1.8]),
    ):
        atomic_npz(
            tmp_path / policy / "uncertainty-evaluation-0.npz",
            identities=np.array([f"ethanol:fixture:{i}" for i in range(4)]),
            errors=error,
        )
        outcomes.append(
            {
                "molecule": "ethanol",
                "trial": 1,
                "budget": 100,
                "policy": policy,
                "round": 0,
                "path": policy,
            }
        )
    data = tmp_path / "author.npz"
    atomic_npz(data, old_indices=pool.original_indices)
    paired_policy_outcomes(tmp_path, outcomes, {"ethanol": pool})
    return tmp_path, outcomes, {"rmd17_ethanol.npz": data}


def test_paired_policies_preserve_geometry_weighting_and_temporal_blocks(policy_arrays):
    root, outcomes, data = policy_arrays
    verify_paired_policies(root, outcomes, data)
    saved = read_json(root / "paired-policy-outcomes.json")
    assert len(saved) == 2 and saved[0]["paired_interval"]["blocks"] == 2


def test_paired_policies_rejects_false_improvement_after_summary_rewrite(policy_arrays):
    root, outcomes, data = policy_arrays
    saved = read_json(root / "paired-policy-outcomes.json")
    saved[0]["paired_interval"]["mean_improvement"] += 1
    write_json(root / "paired-policy-outcomes.json", saved)
    with pytest.raises(IntegrityError, match="retained temporal blocks"):
        verify_paired_policies(root, outcomes, data)


def test_paired_policies_rejects_missing_control_arm(policy_arrays):
    root, outcomes, _ = policy_arrays
    with pytest.raises(ContractError, match="all three controls"):
        paired_policy_outcomes(
            root, outcomes[1:], {"ethanol": SimpleNamespace(original_indices=np.arange(4))}
        )
