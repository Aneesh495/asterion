import io
import struct
import warnings
import zipfile

import numpy as np
import pytest

from asterion.artifacts.io import file_hash
from asterion.contracts.errors import ContractError, ResourceError
from asterion.contracts.units import KCAL_MOL_TO_EV
from asterion.data.archive import dataset_arrays
from asterion.data.dataset import load_md22, load_rmd17


@pytest.fixture
def native():
    return {
        "nuclear_charges": np.array([6, 1]),
        "coords": np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]]
        ),
        "energies": np.array([1.0, 2.0]),
        "forces": np.ones((2, 2, 3)),
        "old_indices": np.array([0, 7]),
    }


def archive(tmp_path, native):
    path = tmp_path / "data.npz"
    np.savez_compressed(path, **native)
    return path


def test_bounded_native_loading_preserves_fortran_coordinates_and_physical_units(tmp_path, native):
    native["coords"] = np.asfortranarray(native["coords"])
    pool, oracle = load_rmd17(archive(tmp_path, native), "native-fixture")
    np.testing.assert_array_equal(pool.positions, native["coords"])
    np.testing.assert_array_equal(pool.original_indices, [0, 7])
    labels = oracle.labels([0], [0], "train")
    assert labels[0].energy == KCAL_MOL_TO_EV
    np.testing.assert_array_equal(labels[0].forces, np.ones((2, 3)) * KCAL_MOL_TO_EV)
    assert not pool.positions.flags.writeable


def test_native_fractional_atomic_numbers_are_rejected_before_integer_cast(tmp_path, native):
    native["nuclear_charges"] = np.array([6.0, 1.5])
    with pytest.raises(ContractError, match="integer vector"):
        load_rmd17(archive(tmp_path, native), "native-fixture")


def test_native_fractional_temporal_indices_are_rejected_before_integer_cast(tmp_path, native):
    native["old_indices"] = np.array([0.0, 7.5])
    with pytest.raises(ContractError, match="temporal indices"):
        load_rmd17(archive(tmp_path, native), "native-fixture")


def test_native_integer_forces_are_rejected_before_unit_conversion(tmp_path, native):
    native["forces"] = np.ones((2, 2, 3), dtype=int)
    with pytest.raises(ContractError, match="energy/force"):
        load_rmd17(archive(tmp_path, native), "native-fixture")


def test_native_transposed_energy_axis_is_not_silently_flattened(tmp_path, native):
    native["energies"] = native["energies"].reshape(1, 2)
    with pytest.raises(ContractError, match="energy/force axes"):
        load_rmd17(archive(tmp_path, native), "native-fixture")


def test_native_integer_coordinates_are_rejected_before_float_cast(tmp_path, native):
    native["coords"] = native["coords"].astype(int)
    with pytest.raises(ContractError, match="coordinate axes or precision"):
        load_rmd17(archive(tmp_path, native), "native-fixture")


def test_archive_rejects_object_array_in_unused_metadata(tmp_path, native):
    native["metadata"] = np.array([{"private": "object-array"}], dtype=object)
    with pytest.raises(ContractError, match="object and structured"):
        load_rmd17(archive(tmp_path, native), "native-fixture")


def test_archive_rejects_duplicate_zip_array_names(tmp_path):
    array = io.BytesIO()
    np.save(array, np.array([1.0]))
    path = tmp_path / "duplicate.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as zipped:
            zipped.writestr("value.npy", array.getvalue())
            zipped.writestr("value.npy", array.getvalue())
    with pytest.raises(ContractError, match="duplicate or unsafe"):
        dataset_arrays(path, {"value"})


def test_archive_rejects_oversized_header_before_allocating_it(tmp_path):
    path = tmp_path / "header.npz"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(
            "value.npy", bytes([147]) + b"NUMPY" + bytes([2, 0]) + struct.pack("<I", 8193)
        )
    with pytest.raises(ResourceError, match="header exceeds"):
        dataset_arrays(path, {"value"})


def test_archive_rejects_large_declared_array_without_allocating_its_body(tmp_path):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (1000000, 1000)}
    )
    path = tmp_path / "large.npz"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("value.npy", header.getvalue())
    with pytest.raises(ResourceError, match="body exceeds"):
        dataset_arrays(path, {"value"})


def test_archive_rejects_truncated_numeric_array_before_loading(tmp_path):
    raw = io.BytesIO()
    np.save(raw, np.zeros(10))
    path = tmp_path / "truncated.npz"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("value.npy", raw.getvalue()[:-8])
    with pytest.raises(ContractError, match="body size"):
        dataset_arrays(path, {"value"})


def test_secondary_native_arrays_remain_immutable_after_loading(tmp_path, native):
    path = tmp_path / "secondary.npz"
    np.savez_compressed(
        path,
        z=native["nuclear_charges"],
        R=native["coords"],
        E=native["energies"][:, None],
        F=native["forces"],
    )
    pool, oracle = load_md22(path, "native-fixture", file_hash(path))
    assert pool.chemistry.level == "PBE+MBD"
    assert not pool.positions.flags.writeable and not pool.numbers.flags.writeable
    np.testing.assert_array_equal(
        oracle.labels([0], [0], "train")[0].forces, np.ones((2, 3)) * KCAL_MOL_TO_EV
    )


def test_loading_rejects_concurrent_author_file_mutation(tmp_path, native, monkeypatch):
    from asterion.data import dataset

    path = archive(tmp_path, native)
    original = dataset.dataset_arrays

    def change_after_read(path, required):
        result = original(path, required)
        with path.open("ab") as stream:
            stream.write(b"concurrent mutation")
        return result

    monkeypatch.setattr(dataset, "dataset_arrays", change_after_read)
    with pytest.raises(ContractError, match="changed during loading"):
        load_rmd17(path, "native-fixture")
