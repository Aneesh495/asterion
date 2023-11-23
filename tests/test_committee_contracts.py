from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool, LabelOracle
from asterion.data.ledger import RevealLedger
from asterion.data.splits import Split
from asterion.dynamics.analytic import HarmonicBond
from asterion.models.potential import ModelConfig, make_model
from asterion.training.checkpoint import load_checkpoint, save_checkpoint
from asterion.training.engine import TrainConfig, train
from asterion.training.identity import state_fingerprint
from asterion.uncertainty.committee import Calibration, Committee


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("trained-committee")
    positions = np.zeros((48, 2, 3))
    positions[:, 1, 0] = np.linspace(0.8, 1.2, 48)
    pool = CoordinatePool(
        np.array([6, 1]), positions, np.arange(48), "analytic", "bond", Chemistry(level="analytic")
    )
    values = [HarmonicBond()(position, pool.numbers) for position in positions]
    oracle = LabelOracle(
        pool, np.array([v.energy for v in values]), np.array([v.forces for v in values])
    )
    split = Split(
        pool.dataset_hash,
        "fixture",
        1,
        tuple(range(12)),
        tuple(range(12, 16)),
        tuple(range(16, 32)),
        tuple(range(32, 48)),
    )
    ledger = RevealLedger(split, directory / "reveal")
    ledger.reveal(range(8), "shared-fixture")
    models, identities, paths = [], [], []
    for seed in (1, 2, 3):
        result = train(
            oracle,
            split,
            directory / f"member-{seed}",
            ModelConfig(width=4, radial=4, depth=1),
            TrainConfig(seed=seed, budget=8, epochs=1, batch_size=4),
            ledger=ledger,
        )
        path = directory / f"member-{seed}" / result["checkpoint"].split("/")[-1]
        model, record = load_checkpoint(path)
        models.append(model)
        identities.append(record["identity"])
        paths.append(path)
    return models, identities, paths, oracle, split


def committee(case):
    models, identities, _, _, _ = case
    return Committee(models, identities, [1, 2, 3])


def calibration():
    return Calibration.fit(
        np.arange(1, 11, dtype=float),
        np.arange(1, 11, dtype=float) * 2,
        0.1,
        ["a", "b", "c"],
        [str(i) for i in range(10)],
    )


def test_committee_uses_actual_saved_member_weights_and_seeds(trained):
    result = committee(trained)
    assert len(set(result.states)) == 3
    assert result.seeds == (1, 2, 3)


def test_committee_captures_member_list_ownership(trained):
    models, identities, _, _, _ = trained
    borrowed = list(models)
    result = Committee(borrowed, identities, [1, 2, 3])
    borrowed.clear()
    assert len(result.models) == 3


def test_committee_rejects_aliasing_one_model_three_times(trained):
    models, identities, _, _, _ = trained
    with pytest.raises(IntegrityError, match="alias"):
        Committee([models[0]] * 3, identities, [1, 2, 3])


def test_committee_rejects_falsely_relabelled_training_seed(trained):
    models, identities, _, _, _ = trained
    with pytest.raises(IntegrityError, match="seed"):
        Committee(models, identities, [4, 5, 6])


def test_committee_rejects_wrong_checkpoint_identity(trained):
    models, identities, _, _, _ = trained
    with pytest.raises(IntegrityError, match="identity"):
        Committee(models, ["wrong", *identities[1:]], [1, 2, 3])


def test_committee_rejects_extra_duplicate_identity_slot(trained):
    models, identities, _, _, _ = trained
    with pytest.raises(ContractError, match="three"):
        Committee(models, [*identities, identities[0]], [1, 2, 3])


def test_committee_rejects_unsaved_untrained_models():
    models = [make_model(ModelConfig(width=4, depth=1, radial=4)).double() for _ in range(3)]
    with pytest.raises(IntegrityError, match="trained"):
        Committee(models, ["a", "b", "c"], [1, 2, 3])


def test_committee_rejects_original_initial_checkpoint_as_trained(trained):
    models, identities, paths, _, _ = trained
    initial, record = load_checkpoint(paths[0].parent / "initial")
    with pytest.raises(IntegrityError, match="trained"):
        Committee([initial, *models[1:]], [record["identity"], *identities[1:]], [1, 2, 3])


def test_committee_rejects_parameter_mutation_after_loading(trained):
    models, identities, _, _, _ = trained
    changed = deepcopy(models[0])
    with torch.no_grad():
        next(changed.parameters()).add_(0.1)
    with pytest.raises(IntegrityError, match="weights changed"):
        Committee([changed, *models[1:]], identities, [1, 2, 3])


def test_committee_rejects_precision_change_after_loading(trained):
    models, identities, _, _, _ = trained
    changed = deepcopy(models[0]).float()
    with pytest.raises(IntegrityError, match="weights changed"):
        Committee([changed, *models[1:]], identities, [1, 2, 3])


def test_committee_rejects_inference_on_changed_member_lifetime(trained):
    models, identities, _, oracle, split = trained
    private = [deepcopy(model) for model in models]
    result = Committee(private, identities, [1, 2, 3])
    with torch.no_grad():
        next(private[0].parameters()).add_(0.1)
    with pytest.raises(IntegrityError, match="lifetime"):
        result.infer([oracle.pool.geometry(split.evaluation[0])])


def test_committee_raw_mean_and_sample_variance_use_three_members(trained):
    _, _, _, oracle, split = trained
    output = committee(trained).infer([oracle.pool.geometry(row) for row in split.evaluation[:2]])
    expected = sum(output.member_forces) / 3
    variance = sum((value - expected) ** 2 for value in output.member_forces) / 2
    np.testing.assert_allclose(output.forces_mean, expected)
    np.testing.assert_allclose(output.force_component_std, np.sqrt(variance))


def test_committee_rejects_duplicate_evaluation_identity(trained):
    _, _, _, oracle, split = trained
    geometry = oracle.pool.geometry(split.evaluation[0])
    with pytest.raises(ContractError, match="repeats"):
        committee(trained).infer([geometry, geometry])


def test_committee_rejects_empty_inference(trained):
    with pytest.raises(ContractError, match="nonempty"):
        committee(trained).infer([])


def test_committee_rejects_zero_batch_size(trained):
    _, _, _, oracle, split = trained
    with pytest.raises(ContractError, match="positive"):
        committee(trained).infer([oracle.pool.geometry(split.evaluation[0])], batch_size=0)


def test_committee_calibration_uses_retained_role_without_caller_forbidden_list(trained):
    _, _, _, oracle, split = trained
    with pytest.raises(IntegrityError, match="held-out"):
        committee(trained).calibrate(
            oracle.labels(split.validation, split.validation, "validation")
        )


def test_committee_calibration_rejects_training_labels_automatically(trained):
    _, _, _, oracle, _ = trained
    with pytest.raises(IntegrityError, match="held-out"):
        committee(trained).calibrate(oracle.labels(range(8), range(8), "train"))


def test_committee_calibration_rejects_final_evaluation_labels(trained):
    _, _, _, oracle, split = trained
    with pytest.raises(IntegrityError, match="held-out"):
        committee(trained).calibrate(
            oracle.labels(split.evaluation, split.evaluation, "evaluation")
        )


def test_committee_calibration_accepts_disjoint_saved_calibration_role(trained):
    _, _, _, oracle, split = trained
    fitted, prediction, errors = committee(trained).calibrate(
        oracle.labels(split.calibration, split.calibration, "calibration")
    )
    assert fitted.sample_count == len(split.calibration)
    assert prediction.force_score.shape == errors.shape == (16,)
    assert np.isfinite(fitted.bounds(prediction.force_score)).all()


def test_calibration_quantile_matches_independent_finite_rank():
    fitted = calibration()
    assert fitted.multiplier == 2
    assert fitted.floor == 0.055
    np.testing.assert_array_equal(fitted.bounds(np.array([1.0, 3.0])), [2.0, 6.0])


def test_calibration_keeps_exact_zero_bound_available_for_perfect_errors():
    fitted = Calibration.fit(
        np.zeros(10), np.zeros(10), 0.1, ["a", "b", "c"], [str(i) for i in range(10)]
    )
    assert fitted.floor == 1e-8 and fitted.multiplier == 0
    np.testing.assert_array_equal(fitted.bounds(np.zeros(3)), np.zeros(3))


def test_calibration_rejects_negative_error_metric():
    with pytest.raises(ContractError, match="values"):
        Calibration.fit(
            np.ones(10), -np.ones(10), 0.1, ["a", "b", "c"], [str(i) for i in range(10)]
        )


def test_calibration_rejects_two_dimensional_geometry_scores():
    with pytest.raises(ContractError, match="geometries"):
        Calibration.fit(
            np.ones((10, 1)), np.ones((10, 1)), 0.1, ["a", "b", "c"], [str(i) for i in range(10)]
        )


def test_calibration_rejects_repeated_held_out_geometry_identity():
    values = asdict(calibration())
    values["identities"] = ["same"] * 10
    with pytest.raises(ContractError, match="unique"):
        Calibration(**values)


def test_calibration_rejects_single_aliased_member_identity():
    values = asdict(calibration())
    values["committee"] = ["a", "a", "a"]
    with pytest.raises(ContractError, match="distinct"):
        Calibration(**values)


def test_calibration_rejects_nonfinite_fitted_multiplier():
    values = asdict(calibration())
    values["multiplier"] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        Calibration(**values)


def test_calibration_rejects_negative_fitted_multiplier():
    values = asdict(calibration())
    values["multiplier"] = -1
    with pytest.raises(ContractError, match="finite"):
        Calibration(**values)


def test_calibration_rejects_zero_stabilizing_floor():
    values = asdict(calibration())
    values["floor"] = 0
    with pytest.raises(ContractError, match="finite"):
        Calibration(**values)


def test_calibration_rejects_inflated_sample_count():
    values = asdict(calibration())
    values["sample_count"] = 11
    with pytest.raises(ContractError, match="identities"):
        Calibration(**values)


def test_calibration_bounds_reject_negative_force_disagreement():
    with pytest.raises(ContractError, match="nonnegative"):
        calibration().bounds(np.array([-1.0]))


def test_calibration_bounds_reject_nonfinite_force_disagreement():
    with pytest.raises(ContractError, match="finite"):
        calibration().bounds(np.array([float("inf")]))


def test_calibration_bounds_reject_scalar_without_geometry_axis():
    with pytest.raises(ContractError, match="vector"):
        calibration().bounds(1.0)


def test_checkpoint_save_rejects_reserved_identity_shadow(model, tmp_path):
    with pytest.raises(IntegrityError, match="reserved"):
        save_checkpoint(tmp_path / "checkpoint", model, None, {"identity": "invented"})
    assert not (tmp_path / "checkpoint").exists()


def test_checkpoint_load_rejects_half_precision_execution(trained):
    with pytest.raises(ContractError, match="audited"):
        load_checkpoint(trained[2][0], dtype=torch.float16)


def test_execution_fingerprint_binds_parameter_precision(trained):
    model = trained[0][0]
    assert state_fingerprint(model) != state_fingerprint(deepcopy(model).float())
