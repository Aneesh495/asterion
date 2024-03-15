"""Package verified evidence with selected weights and a reproducible source checkout."""

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath

from asterion.artifacts.campaign import CampaignOwner
from asterion.artifacts.io import canonical, file_hash, read_json, safe_member, write_json
from asterion.artifacts.source import project_root, source_inputs
from asterion.contracts.errors import IntegrityError, ResourceError

MAX_MEMBERS = 300000
MAX_TOTAL_BYTES = 256 * 1024**3


def archive_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or str(path) != name
        or ".." in path.parts
        or "\\" in name
        or "\x00" in name
        or len(name.encode()) > 1024
    ):
        raise IntegrityError("unsafe release member name")
    return name


def release_inventory(files):
    if not files or len(files) > MAX_MEMBERS or "RELEASE.json" in files:
        raise IntegrityError("empty, oversized or reserved release inventory")
    result = {}
    total = 0
    for name, path in sorted(files.items()):
        archive_name(name)
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("release member is absent or linked")
        size = path.stat().st_size
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ResourceError("release exceeds the declared archive size bound")
        result[name] = {"bytes": size, "sha256": file_hash(path)}
    return result


def tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    return info


def write_archive(path: Path, files, metadata):
    """Deterministic container primitive. Scientific authorization belongs to build_release."""
    inventory = release_inventory(files)
    manifest = {"format": "asterion-release-v1", "metadata": metadata, "files": inventory}
    encoded = canonical(manifest) + b"\n"
    if len(encoded) > 64 * 1024**2:
        raise ResourceError("release manifest exceeds safety bound")
    with path.open("xb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(tar_info("RELEASE.json", len(encoded)), io.BytesIO(encoded))
                for name, record in inventory.items():
                    original = files[name]
                    if original.is_symlink() or original.stat().st_size != record["bytes"]:
                        raise IntegrityError("release input changed before archive write")
                    with original.open("rb") as data:
                        archive.addfile(tar_info(name, record["bytes"]), data)
                    if original.is_symlink() or file_hash(original) != record["sha256"]:
                        raise IntegrityError("release input changed during archive write")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def verify_release_archive(path: Path):
    """Read every payload byte without extracting paths or executing packaged content."""
    observed = set()
    total = 0
    try:
        with (
            gzip.open(path, "rb") as compressed,
            tarfile.open(fileobj=compressed, mode="r|") as archive,
        ):
            first = archive.next()
            if (
                first is None
                or first.name != "RELEASE.json"
                or not first.isfile()
                or first.size > 64 * 1024**2
            ):
                raise IntegrityError("release manifest is absent or unsafe")
            data = archive.extractfile(first)
            if data is None:
                raise IntegrityError("release manifest is unavailable")
            manifest = json.loads(
                data.read(),
                parse_constant=lambda x: (_ for _ in ()).throw(
                    IntegrityError("nonfinite release metadata")
                ),
            )
            if (
                set(manifest) != {"format", "metadata", "files"}
                or manifest["format"] != "asterion-release-v1"
            ):
                raise IntegrityError("unknown release archive format")
            records = manifest["files"]
            if not isinstance(records, dict) or not records or len(records) > MAX_MEMBERS:
                raise IntegrityError("invalid release inventory")
            for name, record in records.items():
                archive_name(name)
                if (
                    name == "RELEASE.json"
                    or set(record) != {"bytes", "sha256"}
                    or type(record["bytes"]) is not int
                    or record["bytes"] < 0
                    or not isinstance(record["sha256"], str)
                    or len(record["sha256"]) != 64
                ):
                    raise IntegrityError("invalid release member identity")
                total += record["bytes"]
            if total > MAX_TOTAL_BYTES:
                raise ResourceError("release exceeds archive size bound")
            while member := archive.next():
                name = archive_name(member.name)
                if (
                    not member.isfile()
                    or name not in records
                    or name in observed
                    or member.size != records[name]["bytes"]
                ):
                    raise IntegrityError(
                        "release has missing, repeated or unexplained member identities"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise IntegrityError("release member is unavailable")
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    size += len(block)
                if size != member.size or digest.hexdigest() != records[name]["sha256"]:
                    raise IntegrityError("release member checksum mismatch")
                observed.add(name)
            if observed != set(records):
                raise IntegrityError("release omitted required members")
            # Tar stops at its end marker before gzip's checksum trailer. Consume that trailer too.
            while compressed.read(1 << 20):
                pass
    except (tarfile.TarError, EOFError, OSError, ValueError, TypeError, KeyError) as error:
        raise IntegrityError(f"invalid release archive: {error}") from error
    return {
        "status": "archive-integrity-verified",
        "sha256": file_hash(path),
        "members": len(observed),
        "payload_bytes": total,
        "metadata": manifest["metadata"],
    }


def publish_archive(output: Path, files, metadata):
    """Retain failed writes and commit the final filename only after a complete readback."""
    if output.exists() or output.is_symlink():
        raise IntegrityError("release output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = output.with_name(f".{output.name}.publication")
    with CampaignOwner(workspace, resume=True):
        attempt = workspace / f"attempt-{uuid.uuid4().hex}"
        attempt.mkdir()
        staged = attempt / "release.tar.gz"
        receipt = {"status": "writing", "output": str(output.resolve()), "metadata": metadata}
        write_json(attempt / "PUBLICATION.json", receipt)
        try:
            write_archive(staged, files, metadata)
            checked = verify_release_archive(staged)
            # A hard link gives exclusive creation on the same filesystem; no existing output is replaced.
            os.link(staged, output)
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            receipt.update(status="published", archive=checked)
            write_json(attempt / "PUBLICATION.json", receipt)
            return {**checked, "path": str(output.resolve())}
        except BaseException as error:
            receipt.update(status="interrupted", error=f"{type(error).__name__}: {error}")
            write_json(attempt / "PUBLICATION.json", receipt)
            raise


def package_command(argv, root, directory, label):
    started = time.time_ns()
    with (
        (directory / f"{label}.stdout").open("wb") as stdout,
        (directory / f"{label}.stderr").open("wb") as stderr,
    ):
        result = subprocess.run(argv, cwd=root, stdout=stdout, stderr=stderr, timeout=600)
    receipt = {
        "argv": argv,
        "started_ns": started,
        "ended_ns": time.time_ns(),
        "exit": result.returncode,
        "stdout_sha256": file_hash(directory / f"{label}.stdout"),
        "stderr_sha256": file_hash(directory / f"{label}.stderr"),
    }
    write_json(directory / f"{label}.json", receipt)
    if result.returncode:
        raise ResourceError(f"release {label} failed; retained logs at {directory}")
    return receipt


def build_release(evidence: Path, storage: Path, output: Path):
    from asterion.artifacts.verify import check_manifest, verify

    root = project_root()
    frozen = source_inputs()
    if frozen["dirty"]:
        raise IntegrityError("release requires committed source inputs")
    if output.exists() or output.is_symlink():
        raise IntegrityError("release output already exists")
    verified = verify(evidence, "full", storage=storage)
    if (
        verified["profile"] != "full"
        or verified["phase"] != "final"
        or verified["status"] != "passed"
    ):
        raise IntegrityError("release requires independently verified full final evidence")
    manifest = check_manifest(evidence)
    if verified["source"] != frozen["digest"] or verified["manifest_sha256"] != file_hash(
        evidence / "MANIFEST.json"
    ):
        raise IntegrityError("release verification differs from frozen source or evidence")
    files = {
        f"evidence/{p}": safe_member(evidence, p) for p in [*manifest["files"], "MANIFEST.json"]
    }
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout
    for raw in tracked.split(b"\0"):
        if raw:
            name = raw.decode()
            files[f"source/{name}"] = safe_member(root, name)
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in files.values()) + 512 * 1024**2
    if shutil.disk_usage(output.parent).free < required:
        raise ResourceError(
            f"release needs at least {required} free bytes for its retained container"
        )
    build_root = output.with_name(f".{output.name}.build")
    with CampaignOwner(build_root, resume=True):
        attempt = build_root / f"attempt-{uuid.uuid4().hex}"
        attempt.mkdir()
        package_command(
            ["uv", "build", "--offline", "--sdist", "--out-dir", str(attempt)],
            root,
            attempt,
            "sdist",
        )
        sdists = list(attempt.glob("*.tar.gz"))
        if len(sdists) != 1:
            raise IntegrityError("release source distribution population differs")
        package_command(
            ["uv", "build", "--offline", "--wheel", "--out-dir", str(attempt), str(sdists[0])],
            root,
            attempt,
            "wheel",
        )
        if len(list(attempt.glob("*.whl"))) != 1:
            raise IntegrityError("release wheel population differs")
        package_command(
            ["git", "bundle", "create", str(attempt / "source.bundle"), "HEAD"],
            root,
            attempt,
            "source-bundle",
        )
        selected = []
        for row in read_json(evidence / "UNITS.json")["training"]:
            directory = safe_member(evidence, row["path"])
            selection = read_json(directory / "selection.json")
            generation = safe_member(directory, selection["checkpoint"])
            identity = read_json(generation / "checkpoint.json")["identity"]
            selected.append(
                {
                    "unit": row,
                    "checkpoint_identity": identity,
                    "generation": str(generation.relative_to(evidence)),
                    "selection_sha256": file_hash(directory / "selection.json"),
                    "files": {
                        str(p.relative_to(evidence)): file_hash(p)
                        for p in generation.rglob("*")
                        if p.is_file()
                    },
                }
            )
        write_json(
            attempt / "SELECTED_MODELS.json", {"source": frozen["digest"], "models": selected}
        )
        write_json(attempt / "VERIFICATION.json", verified)
        write_json(
            attempt / "AUTHOR_DATA.json",
            {
                "primary": read_json(evidence / "DATASETS.json"),
                "secondary": read_json(root / "configs/md22-release.json"),
                "author_bytes_included": False,
            },
        )
        (attempt / "REPRODUCE.md").write_text(
            "# Reproduce a verified release\n\n"
            "Author arrays remain external. Their exact identities and license states are in AUTHOR_DATA.json.\n"
            "Use a compatible Python 3.12 host and the pinned dependencies. CUDA is required to rerun full acceptance.\n\n"
            "```sh\ngit init checkout\ngit -C checkout fetch ../source.bundle HEAD\n"
            "git -C checkout checkout --detach FETCH_HEAD\ncd checkout\nuv sync --locked\n"
            "uv run asterion data --config ../evidence/CAMPAIGN.json --storage storage\n"
            "uv run asterion verify --profile full --config ../evidence/CAMPAIGN.json "
            "--evidence ../evidence --storage storage\n```\n\n"
            "SELECTED_MODELS.json points to the immutable safe tensor generations retained in evidence/.\n"
            "Container readback verifies member hashes. Scientific verification remains the independent command above.\n"
        )
        for p in attempt.iterdir():
            if p.is_file():
                files[p.name] = p
        if (
            source_inputs() != frozen
            or file_hash(evidence / "MANIFEST.json") != verified["manifest_sha256"]
        ):
            raise IntegrityError("release inputs changed during package builds")
        check_manifest(evidence)
        metadata = {
            "kind": "verified-full-release",
            "source": frozen,
            "evidence_manifest_sha256": verified["manifest_sha256"],
            "verification_sha256": file_hash(attempt / "VERIFICATION.json"),
        }
        result = publish_archive(output, files, metadata)
        if (
            source_inputs() != frozen
            or file_hash(evidence / "MANIFEST.json") != verified["manifest_sha256"]
        ):
            # Publication remains inspectable, but this call cannot certify it after a concurrent input change.
            raise IntegrityError("release inputs changed during publication")
        return {**result, "scientific_verification": verified["status"]}
