"""Check embedded viewer observations against raw inputs and the shipped template."""

import json
import re
from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, read_json
from asterion.artifacts.source import project_root
from asterion.artifacts.verify import independent_metrics, read_arrays
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.replay import replay_trajectory


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def viewer_data(path):
    require(path.stat().st_size <= 256 * 1024**2, "viewer exceeds bounded artifact size")
    html = path.read_text()
    match = re.search(r"const data=(.*?);\nconst", html)
    if match is None:
        raise IntegrityError("viewer observation payload is missing")
    template = (project_root() / "src/asterion/viewer/viewer.html").read_text()
    require(
        html == template.replace("__ASTERION_DATA__", match[1]),
        "viewer executable markup differs from the frozen template",
    )
    payload = json.loads(match[1])
    return payload


def verify_viewer(path: Path, predictions: Path, trajectory: Path, expected_acquisition=None):
    payload = viewer_data(path)
    arrays = read_arrays(predictions)
    independent_metrics(arrays)
    replay_trajectory(trajectory)
    metadata = read_json(trajectory / "trajectory.json")
    require(payload["trajectory"] == metadata, "viewer hides trajectory termination or validity")
    require(
        payload["lineage"]["predictions_sha256"] == file_hash(predictions),
        "viewer predictions identity changed",
    )
    require(len(payload["samples"]) == len(arrays["energy"]), "viewer held-out denominator changed")
    for index, sample in enumerate(payload["samples"]):
        begin, end = arrays["pointers"][index : index + 2]
        require(
            all(
                np.array_equal(sample[field], arrays[key][begin:end])
                for field, key in (
                    ("positions", "positions"),
                    ("numbers", "numbers"),
                    ("force", "force"),
                    ("reference", "target_force"),
                )
            )
            and sample["energy"] == float(arrays["energy"][index])
            and sample["target"] == float(arrays["target_energy"][index])
            and sample["identity"] == str(arrays["identities"][index]),
            "viewer held-out observation differs from raw predictions",
        )
    keys = ("potential", "kinetic", "heat", "step", "uncertainty", "uncertainty_available")
    rejected = metadata.get("format") == "asterion-rejected-initial-v1"
    if rejected:
        raw = read_arrays(trajectory / "failure.npz")
        frames = payload["frames"]
        require(
            len(frames) == 1
            and np.array_equal(frames[0]["positions"], raw["positions"])
            and np.array_equal(frames[0]["numbers"], raw["numbers"])
            and all(
                frames[0][key] is None for key in ("force", "reference", "energy", "uncertainty")
            )
            and frames[0]["uncertainty_available"] is False
            and frames[0]["step"] == 0
            and frames[0]["identity"] == "rejected initial state"
            and all(payload["history"][key] == [] for key in keys)
            and payload["lineage"]["rejected_initial_sha256"]
            == file_hash(trajectory / "failure.npz"),
            "viewer invents physical observations for rejected initial evaluation",
        )
    else:
        raw = read_arrays(trajectory / "trajectory.npz")
        stride = max(1, (len(raw["step"]) + 999) // 1000)
        indices = sorted(set(range(0, len(raw["step"]), stride)) | {len(raw["step"]) - 1})
        require(
            payload["lineage"]["trajectory_sha256"] == file_hash(trajectory / "trajectory.npz")
            and len(payload["frames"]) == len(indices)
            and all(np.array_equal(payload["history"][key], raw[key]) for key in keys),
            "viewer trajectory identity, retained history or frame population changed",
        )
        for index, frame in zip(indices, payload["frames"]):
            require(
                np.array_equal(frame["positions"], raw["positions"][index])
                and np.array_equal(frame["numbers"], raw["numbers"])
                and np.array_equal(frame["force"], raw["forces"][index])
                and frame["reference"] is None
                and frame["energy"] == float(raw["potential"][index])
                and frame["step"] == int(raw["step"][index])
                and frame["identity"] == f"step {int(raw['step'][index])}"
                and frame["uncertainty"] == float(raw["uncertainty"][index])
                and frame["uncertainty_available"] is bool(raw["uncertainty_available"][index]),
                "viewer trajectory frame differs from the retained physical observation",
            )
    expected = expected_acquisition or {
        "status": "unavailable",
        "reason": "No acquisition observations attached",
        "rounds": [],
    }
    require(
        payload["acquisition"] == expected, "viewer acquisition observations or identities changed"
    )
    return {
        "samples": len(payload["samples"]),
        "frames": len(payload["frames"]),
        "acquisition_rounds": len(expected["rounds"]),
    }
