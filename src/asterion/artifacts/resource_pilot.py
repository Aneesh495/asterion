"""Independently validate resource-pilot population, states, work and retained outputs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from asterion.artifacts.derivatives import geometry_from_trace
from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.artifacts.runtime import model_observation, timing
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import collate
from asterion.models.potential import ModelConfig
from asterion.training.checkpoint import load_checkpoint
from asterion.training.identity import state_fingerprint


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def verify_resource_pilot(root, path, config, source, datasets):
    report_file = safe_member(root, path)
    directory = report_file.parent
    report = read_json(report_file)
    require(
        report["format"] == "asterion-resource-pilot-v1"
        and report["status"] == "passed"
        and report["source"]["digest"] == source["digest"]
        and report["repetitions"] >= 3
        and report["supervised_labels_consumed"] == 0
        and report["training_or_accuracy_claim"] is False
        and report["full_campaign_hours"] is None,
        "resource pilot source, workload or untrained mechanism scope changed",
    )
    names = list(config["molecules"])
    if "secondary" in config:
        names.append(config["secondary"]["system"])
    expected = {(name, kind) for name in names for kind in config["models"]}
    require(
        len(report["systems"]) == len(expected)
        and {(row["name"], row["kind"]) for row in report["systems"]} == expected,
        "resource pilot omitted or repeated a configured system or model",
    )
    dtype = getattr(torch, config["training"]["dtype"])
    device = "cuda:0" if config["training"]["device"] == "cuda" else config["training"]["device"]
    models = {}
    geometries = {}
    for row in report["systems"]:
        model, checkpoint = load_checkpoint(safe_member(directory, row["path"]), dtype=dtype)
        require(
            checkpoint["identity"] == row["checkpoint"]
            and checkpoint["origin"] == "original-resource-pilot-initialization"
            and checkpoint["seed"] == 491
            and state_fingerprint(model) == row["executed_state"],
            "resource pilot checkpoint or executed state changed",
        )
        raw = [geometry_from_trace(trace) for trace in row["geometries"]]
        expected_architecture = replace(
            ModelConfig(kind=row["kind"], **config["architecture"]), level=raw[0].chemistry.level
        )
        require(
            object_hash(model.config.to_dict())
            == object_hash(expected_architecture.to_dict())
            == object_hash(row["architecture"]),
            "resource pilot architecture differs from the frozen research model",
        )
        require(
            len(raw) == 8 and len({g.identity for g in raw}) == 8,
            "resource pilot geometry work shortened",
        )
        for index, geometry in enumerate(raw):
            name, digest, position = geometry.identity.split(":")
            require(name == row["name"], "resource pilot assigned another system's geometry")
            if row["name"] in config["molecules"]:
                data = datasets[f"rmd17_{name}.npz"]
                coordinate_key, number_key = "coords", "nuclear_charges"
            else:
                manifest = read_json(Path("configs/md22-release.json"))
                data = Path(row["dataset_path"])
                require(
                    file_hash(data) == manifest["sha256"], "pilot secondary author data changed"
                )
                coordinate_key, number_key = "R", "z"
            require(
                digest == file_hash(data) and int(position) == index,
                "pilot source geometry identity changed",
            )
            with np.load(data, allow_pickle=False) as author:
                require(
                    np.array_equal(geometry.positions, author[coordinate_key][index])
                    and np.array_equal(geometry.numbers, author[number_key]),
                    "pilot inputs differ from retained author coordinates",
                )
        key = (row["name"], row["kind"])
        models[key], geometries[key] = model, raw
    expected_samples = {
        (name, kind, size, mode, repetition)
        for name, kind in expected
        for size in (1, 8)
        for mode in ("energy-force", "force-loss-backward")
        for repetition in range(report["repetitions"])
    }
    observed = set()
    for row in report["samples"]:
        key = (row["name"], row["kind"])
        identity = (*key, row["batch_geometries"], row["mode"], row["repetition"])
        require(
            identity in expected_samples and identity not in observed,
            "resource pilot work matrix changed",
        )
        observed.add(identity)
        timing(row)
        require(
            row["dtype"] == str(dtype)
            and row["device"] == device
            and row["atoms"]
            == sum(len(g.numbers) for g in geometries[key][: row["batch_geometries"]])
            and row["collation_timed"] is False
            and row["optimizer_update"] is False
            and type(row["native_peak_rss"]) is int
            and row["native_peak_rss"] > 0
            and type(row["peak_device_bytes"]) is int
            and row["peak_device_bytes"] >= 0
            and (device != "cpu" or row["peak_device_bytes"] == 0),
            "resource pilot backend, memory or mathematical work changed",
        )
        if row["repetition"] == 0:
            model_observation(
                models[key],
                geometries[key][: row["batch_geometries"]],
                {"energy": row["energy"], "forces": row["forces"]},
            )
            if row["mode"] == "force-loss-backward":
                model = models[key]
                model.zero_grad(set_to_none=True)
                batch = collate(geometries[key][: row["batch_geometries"]], dtype=dtype)
                energy, force = model(batch, create_graph=True)
                (force.square().mean() + energy.square().mean() * 1e-6).backward()
                actual = sum(
                    float(p.grad.detach().double().square().sum())
                    for p in model.parameters()
                    if p.grad is not None
                )
                require(
                    np.isfinite(row["gradient_squared_norm"])
                    and np.isclose(
                        row["gradient_squared_norm"],
                        actual,
                        atol=1e-12,
                        rtol=1e-5 if dtype == torch.float32 else 1e-9,
                    ),
                    "resource pilot mixed-gradient observation differs from saved parameters",
                )
        else:
            require(
                row["energy"] is None and row["forces"] is None, "pilot output retention changed"
            )
        if row["mode"] == "force-loss-backward":
            require(
                np.isfinite(row["gradient_squared_norm"]) and row["gradient_squared_norm"] >= 0,
                "resource pilot retained nonfinite mixed gradients",
            )
        else:
            require(
                row["gradient_squared_norm"] is None,
                "forward pilot invents parameter backward work",
            )
    require(observed == expected_samples, "resource pilot omitted required measured repetitions")
    return {"systems": len(expected), "samples": len(observed)}
