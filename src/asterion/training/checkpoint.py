import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from asterion.artifacts.io import file_hash, object_hash, read_json, write_json
from asterion.contracts.errors import ContractError, IntegrityError, ResourceError
from asterion.models.potential import ModelConfig, make_model
from asterion.training.identity import state_fingerprint


def pack_tree(value, tensors: dict, prefix="state"):
    if isinstance(value, torch.Tensor):
        key = f"{prefix}.{len(tensors)}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"tensor": key}
    if isinstance(value, dict):
        return {
            "mapping": [
                [pack_tree(k, tensors, prefix), pack_tree(v, tensors, prefix)]
                for k, v in value.items()
            ]
        }
    if isinstance(value, (list, tuple)):
        return {
            "sequence": [pack_tree(v, tensors, prefix) for v in value],
            "tuple": isinstance(value, tuple),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise IntegrityError(f"unsupported checkpoint type {type(value)}")


def unpack_tree(value, tensors):
    if isinstance(value, dict):
        if "tensor" in value:
            return tensors[value["tensor"]]
        if "mapping" in value:
            return {unpack_tree(k, tensors): unpack_tree(v, tensors) for k, v in value["mapping"]}
        if "sequence" in value:
            items = [unpack_tree(v, tensors) for v in value["sequence"]]
            return tuple(items) if value["tuple"] else items
        raise IntegrityError("invalid checkpoint tree node")
    return value


def save_checkpoint(path: Path, model, optimizer, metadata: dict):
    reserved = {"identity", "architecture", "optimizer", "weights_sha256", "state_sha256", "format"}
    if reserved & metadata.keys():
        raise IntegrityError("checkpoint metadata shadows reserved format fields")
    if path.exists():
        raise IntegrityError("checkpoint generation already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".", dir=path.parent))
    try:
        weights = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
        save_file(weights, str(temporary / "model.safetensors"))
        tensors: dict = {}
        optimizer_tree = pack_tree(optimizer.state_dict() if optimizer is not None else {}, tensors)
        tensors["rng.cpu"] = torch.random.get_rng_state().clone()
        if next(model.parameters()).is_cuda:
            for index, rng in enumerate(torch.cuda.get_rng_state_all()):
                tensors[f"rng.cuda.{index}"] = rng.clone()
        save_file(tensors, str(temporary / "state.safetensors"))
        record = {
            **metadata,
            "architecture": model.config.to_dict(),
            "optimizer": optimizer_tree,
            "weights_sha256": file_hash(temporary / "model.safetensors"),
            "state_sha256": file_hash(temporary / "state.safetensors"),
            "format": "asterion-safe-checkpoint-v1",
        }
        record["identity"] = object_hash(record)
        write_json(temporary / "checkpoint.json", record)
        for file in temporary.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


def load_checkpoint(path: Path, device="cpu", dtype=torch.float64):
    if dtype not in {torch.float32, torch.float64} or torch.device(device).type not in {
        "cpu",
        "cuda",
    }:
        raise ContractError("checkpoint execution requires audited CPU/CUDA float32 or float64")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise ResourceError("checkpoint CUDA execution is unavailable")
    record = read_json(path / "checkpoint.json")
    identity = record.pop("identity")
    if record.get("format") != "asterion-safe-checkpoint-v1" or object_hash(record) != identity:
        raise IntegrityError("checkpoint metadata identity mismatch")
    record["identity"] = identity
    for file, key in (
        ("model.safetensors", "weights_sha256"),
        ("state.safetensors", "state_sha256"),
    ):
        if file_hash(path / file) != record[key]:
            raise IntegrityError("checkpoint tensor hash mismatch")
    config = dict(record["architecture"])
    config["elements"] = tuple(config["elements"])
    if config.get("composition") is not None:
        config["composition"] = tuple(config["composition"])
    with torch.random.fork_rng(devices=[]):
        model = make_model(ModelConfig(**config)).to(device=device, dtype=dtype)
    model.load_state_dict(
        load_file(str(path / "model.safetensors"), device=str(device)), strict=True
    )
    boundary = None
    if record.get("origin") in {"original-force-loss-training", "original-random-initialization"}:
        owner = path.parent
        training = read_json(owner / "config.json")
        if object_hash(training) != record["lineage"]:
            raise IntegrityError("checkpoint training owner lineage mismatch")
        split = read_json(owner / "split.json")
        normalization = read_json(owner / "normalization.json")
        prefix = normalization["fit_identities"][0].rsplit(":", 1)[0]
        boundary = {
            "fit": tuple(normalization["fit_identities"]),
            "validation": tuple(f"{prefix}:{row}" for row in split["validation"]),
            "calibration": tuple(f"{prefix}:{row}" for row in split["calibration"]),
        }
    model._asterion_provenance = {
        "checkpoint": record,
        "execution_state": state_fingerprint(model),
        "boundary": boundary,
    }
    return model, record


def restore_optimizer(path: Path, optimizer, record):
    tensors = load_file(str(path / "state.safetensors"))
    optimizer.load_state_dict(unpack_tree(record["optimizer"], tensors))
    torch.random.set_rng_state(tensors["rng.cpu"])
    cuda = sorted(
        (int(key.rsplit(".", 1)[1]), value)
        for key, value in tensors.items()
        if key.startswith("rng.cuda.")
    )
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise IntegrityError("CUDA recovery device population changed")
        torch.cuda.set_rng_state_all([value for _, value in cuda])
