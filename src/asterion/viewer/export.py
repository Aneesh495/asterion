import json
from pathlib import Path

import numpy as np

from asterion.artifacts.io import atomic_bytes, file_hash, read_json
from asterion.contracts.errors import IntegrityError


def export_viewer(predictions: Path, trajectory: Path, output: Path, lineage: dict):
    from asterion.artifacts.verify import independent_metrics
    from asterion.evaluation.replay import replay_trajectory

    replay_trajectory(trajectory)
    with np.load(predictions, allow_pickle=False) as raw:
        a = {key: raw[key] for key in raw.files}
    with np.load(trajectory / "trajectory.npz", allow_pickle=False) as raw:
        t = {key: raw[key] for key in raw.files}
    if not len(a["energy"]) or not len(t["positions"]):
        raise IntegrityError("viewer requires real nonempty predictions and trajectory")
    independent_metrics(a)
    samples = []
    for i, (begin, end) in enumerate(zip(a["pointers"][:-1], a["pointers"][1:])):
        samples.append(
            {
                "positions": a["positions"][begin:end].tolist(),
                "numbers": a["numbers"][begin:end].tolist(),
                "force": a["force"][begin:end].tolist(),
                "reference": a["target_force"][begin:end].tolist(),
                "energy": float(a["energy"][i]),
                "target": float(a["target_energy"][i]),
                "identity": str(a["identities"][i]),
            }
        )
    stride = max(1, (len(t["positions"]) + 999) // 1000)
    indices = sorted(set(range(0, len(t["positions"]), stride)) | {len(t["positions"]) - 1})
    frames = [
        {
            "positions": t["positions"][i].tolist(),
            "numbers": t["numbers"].tolist(),
            "force": t["forces"][i].tolist(),
            "reference": None,
            "energy": float(t["potential"][i]),
            "identity": f"step {int(t['step'][i])}",
            "uncertainty": float(t["uncertainty"][i]),
            "uncertainty_available": bool(t["uncertainty_available"][i]),
            "step": int(t["step"][i]),
        }
        for i in indices
    ]
    metadata = read_json(trajectory / "trajectory.json")
    payload = {
        "samples": samples,
        "frames": frames,
        "trajectory": metadata,
        "lineage": {
            **lineage,
            "predictions_sha256": file_hash(predictions),
            "trajectory_sha256": metadata["trajectory_sha256"],
        },
        "history": {
            key: t[key].tolist()
            for key in (
                "potential",
                "kinetic",
                "heat",
                "step",
                "uncertainty",
                "uncertainty_available",
            )
        },
    }
    template = Path(__file__).with_name("viewer.html").read_text()
    serialized = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    atomic_bytes(output, template.replace("__ASTERION_DATA__", serialized).encode())
    return output
