import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.runtime import verify_runtime
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import Geometry
from asterion.evaluation.benchmark import benchmark
from asterion.models.potential import ModelConfig, make_model


@pytest.fixture(scope="module", params=["invariant", "equivariant"])
def measured(request, tmp_path_factory):
    torch.manual_seed(491)
    model = make_model(ModelConfig(kind=request.param, width=2, depth=1, radial=3)).double()
    geometry = Geometry(
        np.array([6, 1, 8]),
        np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.0, -0.4, 1.3]]),
        "runtime",
    )
    root = tmp_path_factory.mktemp("runtime")
    benchmark(
        model, [geometry], root / "benchmark.json", repetitions=2, long_repetitions=2, long_steps=4
    )
    return root, model, geometry


@pytest.fixture
def observation(measured, tmp_path):
    root, model, geometry = measured
    shutil.copytree(root, tmp_path / "observations")
    root = tmp_path / "observations"
    return root, model, geometry, read_json(root / "benchmark.json")


def verify(observation):
    root, model, _, report = observation
    write_json(root / "benchmark.json", report)
    return verify_runtime(root, "benchmark.json", model)


def sample(observation, mode):
    return next(row for row in observation[3]["samples"] if row["mode"] == mode)


def test_runtime_verifier_replays_curvature_and_actual_dynamics(observation):
    assert verify(observation) == {
        "short_samples": 48,
        "long_samples": 4,
        "completed_dynamics_steps": 8,
    }


def test_runtime_rejects_duplicate_matrix_cell(observation):
    observation[3]["samples"].append(observation[3]["samples"][0])
    with pytest.raises(IntegrityError, match="work identities"):
        verify(observation)


def test_runtime_rejects_missing_transfer_cell(observation):
    observation[3]["samples"].remove(sample(observation, "host_device_transfer"))
    with pytest.raises(IntegrityError, match="omits"):
        verify(observation)


def test_runtime_rejects_relabelled_energy_only_obligation(observation):
    observation[3]["timing_obligation"] = "energy-only"
    with pytest.raises(IntegrityError, match="obligation"):
        verify(observation)


def test_runtime_rejects_false_duration(observation):
    sample(observation, "energy_force")["seconds"] *= 2
    with pytest.raises(IntegrityError, match="duration"):
        verify(observation)


def test_runtime_rejects_overlapping_clock_intervals(observation):
    first, second = observation[3]["samples"][:2]
    second["started_ns"], second["ended_ns"] = first["started_ns"], first["ended_ns"]
    second["seconds"] = first["seconds"]
    with pytest.raises(IntegrityError, match="overlap"):
        verify(observation)


def test_runtime_rejects_switched_sample_precision(observation):
    sample(observation, "force_loss_backward")["dtype"] = "torch.float32"
    with pytest.raises(IntegrityError, match="precision"):
        verify(observation)


def test_runtime_rejects_changed_geometry_input(observation):
    observation[3]["geometries"][0]["positions"][1][0] += 0.1
    with pytest.raises(IntegrityError, match="weights"):
        verify(observation)


def test_runtime_rejects_false_batch_atom_count(observation):
    sample(observation, "graph")["atoms"] += 1
    with pytest.raises(IntegrityError, match="batch scaling"):
        verify(observation)


def test_runtime_rejects_false_graph_edge_work(observation):
    sample(observation, "graph")["edges"] += 1
    with pytest.raises(IntegrityError, match="edge work"):
        verify(observation)


def test_runtime_rejects_cache_rebuild_hidden_as_reuse(observation):
    sample(observation, "cached_graph")["cache_rebuilt"] = False
    with pytest.raises(IntegrityError, match="neighbor reuse"):
        verify(observation)


def test_runtime_rejects_invented_cpu_device_memory(observation):
    sample(observation, "energy_force")["peak_device_allocated_bytes"] = 1
    with pytest.raises(IntegrityError, match="memory"):
        verify(observation)


def test_runtime_rejects_substituted_force_observation(observation):
    sample(observation, "force_loss_backward")["observation"]["forces"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="weights"):
        verify(observation)


def test_runtime_rejects_substituted_curvature_observation(observation):
    sample(observation, "hessian_vector")["observation"]["hessian_product"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="curvature"):
        verify(observation)


def test_runtime_rejects_shortened_long_inference(observation):
    sample(observation, "long_energy_force")["completed_calls"] = 1
    with pytest.raises(IntegrityError, match="shortened"):
        verify(observation)


def test_runtime_rejects_inflated_dynamics_steps(observation):
    sample(observation, "original_dynamics")["completed_steps"] += 10
    with pytest.raises(IntegrityError, match="completed work"):
        verify(observation)


def test_runtime_rejects_false_dynamics_throughput(observation):
    sample(observation, "original_dynamics")["completed_steps_per_second"] *= 2
    with pytest.raises(IntegrityError, match="throughput"):
        verify(observation)


def test_runtime_rejects_escaped_trajectory_path(observation):
    sample(observation, "original_dynamics")["path"] = "../unrelated"
    with pytest.raises(IntegrityError, match="path"):
        verify(observation)


def test_runtime_rejects_inference_labelled_as_dynamics(observation):
    sample(observation, "original_dynamics")["includes_serialization"] = False
    with pytest.raises(IntegrityError, match="end-to-end"):
        verify(observation)


def test_runtime_rejects_another_executed_parameter_state(observation):
    observation[3]["executed_state"] = "0" * 64
    with pytest.raises(IntegrityError, match="parameter"):
        verify(observation)


def test_runtime_rejects_short_diagnostic_as_full_acceptance(observation):
    root, model, _, _ = observation
    with pytest.raises(IntegrityError, match="shortened"):
        verify_runtime(root, "benchmark.json", model, full=True)


def test_benchmark_refuses_to_overwrite_measured_work(measured):
    root, model, geometry = measured
    with pytest.raises(IntegrityError, match="already exists"):
        benchmark(model, [geometry], root / "benchmark.json", repetitions=1)


def test_benchmark_rejects_duplicate_geometry_identity(measured, tmp_path):
    _, model, geometry = measured
    with pytest.raises(ContractError, match="unique"):
        benchmark(model, [geometry, geometry], tmp_path / "bench.json", repetitions=1)
    assert not tmp_path.joinpath("bench.json").exists()


def test_benchmark_rejects_single_atom_dynamics_before_execution(measured, tmp_path):
    _, model, geometry = measured
    single = replace(geometry, numbers=np.array([1]), positions=np.zeros((1, 3)))
    with pytest.raises(ContractError, match="two atoms"):
        benchmark(model, [single], tmp_path / "bench.json", repetitions=1)
    assert not tmp_path.joinpath("benchmark-dynamics").exists()
