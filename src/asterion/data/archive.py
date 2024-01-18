"""Bound array allocation from ZIP inventories and NPY headers before reading bodies."""

import io
import math
import struct
import zipfile

import numpy as np

from asterion.contracts.errors import ContractError, ResourceError


def dataset_size(path):
    if path.stat().st_size > 512 * 1024**2:
        raise ResourceError("compressed dataset exceeds file budget")


def array_header(stream):
    magic = stream.read(8)
    if magic not in {b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00"}:
        raise ContractError("dataset requires audited NPY v1 or v2 array format")
    size_bytes = 2 if magic[-2:] == b"\x01\x00" else 4
    length_raw = stream.read(size_bytes)
    if len(length_raw) != size_bytes:
        raise ContractError("truncated dataset array header")
    length = struct.unpack("<H" if size_bytes == 2 else "<I", length_raw)[0]
    if not 1 <= length <= 8192:
        raise ResourceError("dataset array header exceeds bounded metadata size")
    header = stream.read(length)
    if len(header) != length:
        raise ContractError("truncated dataset array header")
    bounded = io.BytesIO(length_raw + header)
    reader = (
        np.lib.format.read_array_header_1_0
        if size_bytes == 2
        else np.lib.format.read_array_header_2_0
    )
    shape, fortran, dtype = reader(bounded)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ContractError("object and structured dataset arrays are unsupported")
    if dtype.kind not in "biufUS" or not 1 <= dtype.itemsize <= 256:
        raise ContractError("dataset array has unsupported native dtype")
    if len(shape) > 4 or any(type(axis) is not int or not 0 <= axis <= 1000000 for axis in shape):
        raise ResourceError("dataset array axes exceed the declared allocation budget")
    size = math.prod(shape) * dtype.itemsize
    if size > 512 * 1024**2:
        raise ResourceError("dataset array body exceeds the declared allocation budget")
    return shape, fortran, dtype, size, 8 + size_bytes + length


def dataset_arrays(path, required):
    dataset_size(path)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                not 1 <= len(entries) <= 128
                or sum(entry.file_size for entry in entries) > 512 * 1024**2
            ):
                raise ResourceError("expanded dataset archive exceeds member or byte budget")
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or any(
                not name.endswith(".npy") or "/" in name or "\\" in name or ".." in name
                for name in names
            ):
                raise ContractError("dataset archive has duplicate or unsafe array names")
            if not {f"{key}.npy" for key in required} <= set(names):
                raise ContractError("dataset array keys missing")
            headers = {}
            for entry in entries:
                if entry.flag_bits & 1 or entry.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise ContractError("encrypted or unsupported dataset compression")
                with archive.open(entry) as stream:
                    record = array_header(stream)
                if record[3] + record[4] != entry.file_size:
                    raise ContractError("dataset array header disagrees with retained body size")
                headers[entry.filename] = record
            result = {}
            for key in required:
                with archive.open(f"{key}.npy") as stream:
                    shape, fortran, dtype, size, offset = array_header(stream)
                    if (shape, fortran, dtype, size, offset) != headers[f"{key}.npy"]:
                        raise ContractError("dataset array header changed during bounded read")
                    body = stream.read(size + 1)
                    if len(body) != size:
                        raise ContractError("dataset array body is truncated or oversized")
                value = np.frombuffer(body, dtype=dtype).reshape(
                    shape, order="F" if fortran else "C"
                )
                result[key] = value
            return result
    except (zipfile.BadZipFile, EOFError, ValueError, struct.error) as error:
        raise ContractError(f"malformed dataset archive: {error}") from error


def validate_native_arrays(numbers, coordinates, energy, force, indices=None):
    if numbers.ndim != 1 or numbers.dtype.kind not in "iu" or not 0 < len(numbers) <= 512:
        raise ContractError("native dataset atomic numbers require a bounded integer vector")
    if not set(numbers.tolist()) <= {1, 6, 7, 8}:
        raise ContractError("native dataset chemistry is unsupported")
    if (
        coordinates.dtype.kind != "f"
        or coordinates.dtype.itemsize not in {4, 8}
        or coordinates.ndim != 3
        or coordinates.shape[1:] != (len(numbers), 3)
        or not 0 < len(coordinates) <= 250000
    ):
        raise ContractError("native dataset coordinate axes or precision are invalid")
    if (
        energy.dtype.kind != "f"
        or energy.dtype.itemsize not in {4, 8}
        or energy.shape not in {(len(coordinates),), (len(coordinates), 1)}
        or force.dtype.kind != "f"
        or force.dtype.itemsize not in {4, 8}
        or force.shape != coordinates.shape
    ):
        raise ContractError("native dataset energy/force axes or precision are invalid")
    if not all(np.isfinite(value).all() for value in (coordinates, energy, force)):
        raise ContractError("nonfinite native dataset physical values")
    if indices is not None and (
        indices.dtype.kind not in "iu"
        or indices.shape != (len(coordinates),)
        or np.any(indices < 0)
        or np.any(indices > 10000000)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ContractError("native temporal indices require a unique nonnegative integer vector")
