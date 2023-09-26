from dataclasses import replace

import numpy as np
import pytest
import torch
from e3nn import o3

from asterion.contracts.errors import ContractError, IntegrityError, ResourceError
from asterion.contracts.geometry import Chemistry, collate
from asterion.data.splits import Split
from asterion.models.potential import ModelConfig, make_model
from asterion.representations.features import ParityGate, TensorFeature
from asterion.training.engine import TrainConfig, train


def test_batch_retains_electronic_level(geometry):
    changed = replace(geometry, chemistry=Chemistry(level="PBE+MBD"))
    assert collate([changed]).chemistry.level == "PBE+MBD"


def test_geometry_rejects_unknown_electronic_level(geometry):
    with pytest.raises(ContractError, match="level"):
        replace(geometry, chemistry=Chemistry(level="unrecognized-method")).validate()


def test_analytic_task_is_explicitly_separate(geometry):
    batch = collate([replace(geometry, chemistry=Chemistry(level="analytic"))])
    with pytest.raises(ContractError, match="levels differ"):
        make_model(ModelConfig(width=4, depth=1, radial=4)).double()(batch)


def test_model_rejects_wrong_electronic_level(geometry):
    batch = collate([replace(geometry, chemistry=Chemistry(level="PBE+MBD"))])
    with pytest.raises(ContractError, match="levels differ"):
        make_model(ModelConfig(width=4, depth=1, radial=4)).double()(batch)


def test_trained_composition_rejects_different_atom_counts(geometry):
    with pytest.raises(ContractError, match="composition"):
        make_model(ModelConfig(width=4, depth=1, radial=4, composition=(2, 1, 0, 0))).double()(
            collate([geometry])
        )


def test_trained_composition_supports_batched_identical_molecules(geometry):
    composition = tuple(int(np.count_nonzero(geometry.numbers == z)) for z in (1, 6, 7, 8))
    model = make_model(ModelConfig(width=4, depth=1, radial=4, composition=composition)).double()
    energy, force = model(collate([geometry, geometry]))
    assert torch.allclose(energy[0], energy[1])
    assert force.shape == (2 * len(geometry.numbers), 3)


def test_model_resource_budget_rejects_excess_width():
    with pytest.raises(ResourceError, match="size budget"):
        make_model(ModelConfig(width=100000))


def test_model_resource_budget_rejects_excess_depth():
    with pytest.raises(ResourceError, match="size budget"):
        make_model(ModelConfig(depth=100000))


def test_model_rejects_nonfinite_cutoff():
    with pytest.raises(ContractError, match="finite"):
        ModelConfig(cutoff=float("inf")).validate()


def test_model_rejects_nonfinite_normalization():
    with pytest.raises(ContractError, match="finite"):
        ModelConfig(expected_neighbors=float("nan")).validate()


def test_model_rejects_nonfinite_energy_offset():
    with pytest.raises(ContractError, match="finite"):
        ModelConfig(energy_offset=float("nan")).validate()


def test_model_rejects_duplicate_element_embeddings():
    with pytest.raises(ContractError, match="duplicate"):
        ModelConfig(elements=(1, 1, 6, 8)).validate()


def test_model_rejects_zero_activation_budget():
    with pytest.raises(ResourceError, match="activation budget"):
        ModelConfig(max_message_elements=0).validate()


def test_model_stops_before_oversized_tensor_products(geometry):
    model = make_model(ModelConfig(width=4, depth=1, radial=4, max_message_elements=1)).double()
    with pytest.raises(ResourceError, match="message budget"):
        model(collate([geometry]))


def test_tensor_feature_rejects_untracked_geometry_identity():
    with pytest.raises(ContractError, match="identity"):
        TensorFeature(torch.zeros(1, 1, dtype=torch.float64), o3.Irreps("0e"), ("",)).validate(1)


def test_tensor_feature_rejects_physical_unit_on_hidden_values():
    with pytest.raises(ContractError, match="units"):
        TensorFeature(
            torch.zeros(1, 1, dtype=torch.float64), o3.Irreps("0e"), ("frame",), "eV"
        ).validate(1)


def test_tensor_feature_rejects_wrong_representation_dimension():
    with pytest.raises(ContractError, match="axes"):
        TensorFeature(torch.zeros(1, 2, dtype=torch.float64), o3.Irreps("1o"), ("frame",)).validate(
            1
        )


def test_tensor_feature_rejects_half_precision_contract():
    with pytest.raises(ContractError, match="precision"):
        TensorFeature(torch.zeros(1, 1, dtype=torch.float16), o3.Irreps("0e"), ("frame",)).validate(
            1
        )


def test_parity_gate_rejects_odd_scalar_control():
    with pytest.raises(ContractError, match="even scalar"):
        ParityGate(o3.Irreps("1x0o+1x1o"), 1)


def test_real_interactions_return_typed_features(model, geometry):
    outputs = []
    handles = [
        block.register_forward_hook(lambda module, args, output: outputs.append(output))
        for block in model.blocks
    ]
    try:
        model(collate([geometry]))
    finally:
        for handle in handles:
            handle.remove()
    assert len(outputs) == model.config.depth
    assert all(
        isinstance(output, TensorFeature)
        and output.graph_identity == (geometry.identity,)
        and output.unit == "dimensionless"
        for output in outputs
    )


def test_typed_features_preserve_force_loss_gradients(model, geometry):
    _, force = model(collate([geometry]), create_graph=True)
    gradients = torch.autograd.grad(
        force.square().sum(), tuple(model.parameters()), allow_unused=True
    )
    assert sum(g.square().sum().item() for g in gradients if g is not None) > 1e-20


def test_training_rejects_dataset_mismatch_before_label_access(tmp_path):
    split = Split("original-dataset", "fixture", 1, (0, 1, 2, 3), (4, 5), (6, 7), (8, 9))

    class Pool:
        dataset_hash = "different-dataset"

        def __len__(self):
            return 100000

    class Oracle:
        pool = Pool()

        def labels(self, *args):
            raise AssertionError("labels were accessed")

    with pytest.raises(IntegrityError, match="another dataset"):
        train(Oracle(), split, tmp_path / "train", ModelConfig(), TrainConfig(budget=4, epochs=1))
    assert not (tmp_path / "train").exists()
