"""Freeze the consumed configuration and recheck external bytes around managed stages."""

from pathlib import Path

from asterion.artifacts.io import file_hash, read_json, write_json
from asterion.contracts.errors import IntegrityError


class CampaignInputs:
    def __init__(self, run: Path, config, storage: Path, source: str):
        self.run, self.storage, self.config = run, storage.resolve(), config
        self.configuration = run / "FROZEN_CONFIGURATION.json"
        self.identity = run / "INPUT_IDENTITIES.json"
        if self.configuration.exists():
            if read_json(self.configuration) != config:
                raise IntegrityError(
                    "frozen campaign configuration differs from the resumed request"
                )
        else:
            write_json(self.configuration, config)
        expected = {
            "format": "asterion-campaign-inputs-v1",
            "source": source,
            "configuration_sha256": file_hash(self.configuration),
            "configuration_path": str(self.configuration.resolve()),
            "storage": str(self.storage),
        }
        if self.identity.exists():
            self.record = read_json(self.identity)
            if {key: self.record[key] for key in expected} != expected:
                raise IntegrityError("campaign input identity changed during continuation")
        else:
            self.record = {**expected, "raw_files": None}
            write_json(self.identity, self.record)

    def freeze_data(self):
        from asterion.artifacts.datasets import checked_md22
        from asterion.artifacts.verify import check_dataset

        primary = check_dataset(read_json(self.storage / "rmd17/DATASETS.json"), self.storage)
        paths = {f"rmd17/{name}": path for name, path in primary.items()}
        if "secondary" in self.config:
            record = read_json(self.storage / "md22/DATASETS.json")
            path = checked_md22(record, self.storage)
            paths[f"md22/{record['filename']}"] = path
        observed = {
            name: {"sha256": file_hash(path), "size": path.stat().st_size}
            for name, path in paths.items()
        }
        if self.record["raw_files"] is not None and self.record["raw_files"] != observed:
            raise IntegrityError("author input population changed since the frozen data stage")
        self.record["raw_files"] = observed
        write_json(self.identity, self.record)

    def check(self, require_data=True):
        if read_json(self.identity) != self.record:
            raise IntegrityError("campaign input identity record changed")
        if (
            file_hash(self.configuration) != self.record["configuration_sha256"]
            or read_json(self.configuration) != self.config
        ):
            raise IntegrityError("consumed frozen campaign configuration changed")
        files = self.record["raw_files"]
        if require_data and files is None:
            raise IntegrityError("author data identities are unavailable before this stage")
        for relative, expected in (files or {}).items():
            path = self.storage / relative
            if (
                not path.is_file()
                or path.stat().st_size != expected["size"]
                or file_hash(path) != expected["sha256"]
            ):
                raise IntegrityError(f"frozen author bytes changed: {relative}")


def verify_input_identities(
    root, source, config, primary_paths, storage=None, finals=None, secondary_record=None
):
    from asterion.artifacts.datasets import checked_md22

    record = read_json(root / "INPUT_IDENTITIES.json")
    if (
        record["format"] != "asterion-campaign-inputs-v1"
        or record["source"] != source["digest"]
        or record["configuration_sha256"] != file_hash(root / "FROZEN_CONFIGURATION.json")
        or read_json(root / "FROZEN_CONFIGURATION.json") != config
    ):
        raise IntegrityError("full campaign frozen configuration or source lineage changed")
    expected = {
        f"rmd17/{name}": {"sha256": file_hash(path), "size": path.stat().st_size}
        for name, path in primary_paths.items()
    }
    if "secondary" in config:
        secondary = (
            secondary_record or read_json(root / "gates/secondary/secondary.json")["dataset"]
        )
        path = checked_md22(secondary, storage)
        expected[f"md22/{secondary['filename']}"] = {
            "sha256": file_hash(path),
            "size": path.stat().st_size,
        }
    if record["raw_files"] != expected:
        raise IntegrityError("full campaign frozen author population differs from pinned bytes")
    if finals is not None:
        for name, receipt in finals.items():
            if name == "tests":
                continue
            argv = receipt["argv"]
            if (
                argv[argv.index("--config") + 1] != record["configuration_path"]
                or argv[argv.index("--storage") + 1] != record["storage"]
            ):
                raise IntegrityError("managed stage did not consume the frozen campaign inputs")
    return record
