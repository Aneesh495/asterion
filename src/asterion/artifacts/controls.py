import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from asterion.artifacts.io import atomic_npz, file_hash, object_hash, read_json, write_json
from asterion.contracts.errors import IntegrityError


def refresh_manifest(root: Path):
    files = {
        str(path.relative_to(root)): file_hash(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    write_json(root / "MANIFEST.json", {"format": "asterion-evidence-v1", "files": files})


def modify_json(path, mutation):
    value = read_json(path)
    mutation(value)
    write_json(path, value)


def controls(bundle: Path, output: Path, profile="diagnostic"):
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    command = [
        sys.executable,
        "-m",
        "asterion.cli",
        "verify",
        "--evidence",
        str(bundle),
        "--profile",
        profile,
    ]
    positive = subprocess.run(command, capture_output=True, text=True, env=env)
    if positive.returncode != 0:
        raise IntegrityError(f"positive bundle failed before controls: {positive.stderr}")
    results = []
    names = (
        "remove-raw",
        "alter-byte",
        "other-source-commit",
        "change-split",
        "zero-count",
        "inflate-count",
        "remove-baseline",
        "hide-seed",
        "untrained-checkpoint",
        "corrupt-prediction",
        "shorten-workload",
        "omit-source-inputs",
        "swapped-force-units",
        "alter-reveal",
        "false-summary",
        "detached-force-loss",
    )
    for name in names:
        with tempfile.TemporaryDirectory(prefix="asterion-control-") as temporary:
            root = Path(temporary) / "bundle"
            shutil.copytree(bundle, root)
            units = read_json(root / "UNITS.json")
            run = root / units["training"][0]["path"]
            prediction = run / "evaluation-predictions.npz"
            semantic = name not in {"remove-raw", "alter-byte", "omit-source-inputs"}
            if name == "remove-raw":
                prediction.unlink()
            elif name == "alter-byte":
                with prediction.open("ab") as stream:
                    stream.write(b"corrupt")
            elif name == "other-source-commit":
                modify_json(root / "SOURCE_INPUTS.json", lambda v: v.update(commit="0" * 40))
            elif name == "change-split":

                def mutate(v):
                    v["evaluation"][0] = v["train_pool"][0]
                    v["digest"] = object_hash({k: x for k, x in v.items() if k != "digest"})

                modify_json(run / "split.json", mutate)
            elif name == "zero-count":
                modify_json(root / "ACCEPTANCE.json", lambda v: v.update(completed_training_runs=0))
            elif name == "inflate-count":
                modify_json(
                    root / "ACCEPTANCE.json",
                    lambda v: v.update(completed_learned_steps=v["completed_learned_steps"] + 1000),
                )
            elif name == "remove-baseline":
                units["training"] = [
                    entry for entry in units["training"] if entry["model"] != "invariant"
                ]
                write_json(root / "UNITS.json", units)
            elif name == "hide-seed":
                units["training"] = units["training"][1:]
                write_json(root / "UNITS.json", units)
            elif name == "untrained-checkpoint":
                selection = read_json(run / "selection.json")
                checkpoint = run / selection["checkpoint"]
                shutil.copyfile(
                    run / "initial" / "model.safetensors", checkpoint / "model.safetensors"
                )
                record = read_json(checkpoint / "checkpoint.json")
                record["weights_sha256"] = file_hash(checkpoint / "model.safetensors")
                record["identity"] = object_hash(
                    {k: v for k, v in record.items() if k != "identity"}
                )
                write_json(checkpoint / "checkpoint.json", record)
                modify_json(
                    run / "evaluation-metrics.json",
                    lambda v: v.update(checkpoint=record["identity"]),
                )
            elif name in {"corrupt-prediction", "swapped-force-units"}:
                with np.load(prediction, allow_pickle=False) as raw:
                    arrays = {key: raw[key] for key in raw.files}
                if name == "corrupt-prediction":
                    arrays["force"] += 0.25
                else:
                    arrays["target_force"] *= 23.0605
                atomic_npz(prediction, **arrays)
            elif name == "shorten-workload":
                modify_json(run / "history.json", lambda v: v.pop())
            elif name == "omit-source-inputs":
                (root / "SOURCE_INPUTS.json").unlink()
            elif name == "alter-reveal":
                path = run / "reveal" / "round-0000.json"

                def mutate(v):
                    v["rows"][0] = read_json(run / "split.json")["evaluation"][0]
                    v["digest"] = object_hash({k: x for k, x in v.items() if k != "digest"})

                modify_json(path, mutate)
            elif name == "false-summary":
                modify_json(
                    run / "evaluation-metrics.json",
                    lambda v: v["trained"].update(force_mae_eV_angstrom=0.000001),
                )
            elif name == "detached-force-loss":
                modify_json(run / "history.json", lambda v: v[0].update(force_gradient_squared=0.0))
            if semantic:
                refresh_manifest(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "asterion.cli",
                    "verify",
                    "--evidence",
                    str(root),
                    "--profile",
                    profile,
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode == 0:
                raise IntegrityError(f"verifier accepted corrupted bundle: {name}")
            results.append(
                {
                    "control": name,
                    "semantic_rehashed": semantic,
                    "exit": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            print(f"negative control rejected: {name}", flush=True)
    after = subprocess.run(command, capture_output=True, text=True, env=env)
    if after.returncode != 0:
        raise IntegrityError("valid bundle failed positive verification after controls")
    record = {
        "profile": profile,
        "controls": results,
        "positive_before": {"exit": positive.returncode, "stdout": positive.stdout},
        "positive_after": {"exit": after.returncode, "stdout": after.stdout},
        "source_bundle_unchanged": True,
    }
    write_json(output, record)
    return record
