import numpy as np
import pytest

from asterion.contracts.errors import ContractError, ResourceError
from asterion.data.extxyz import reference_arrays

FRAME = (
    "1\nProperties=species:S:1:pos:R:3:REF_forces:R:3:PRED_forces:R:3 "
    'REF_energy=-1.5 PRED_energy=-1.4 asterion_id="fixture:1" pbc="F F F"\n'
    "C 0.1 0.2 0.3 1 2 3 1.1 2.1 3.1\n"
)


def arrays(tmp_path, contents=FRAME, **limits):
    path = tmp_path / "observations.xyz"
    path.write_text(contents)
    return reference_arrays(path, **limits)


def test_reference_parser_preserves_physical_coordinates(tmp_path):
    result = arrays(tmp_path)
    assert np.array_equal(result["positions"], [[0.1, 0.2, 0.3]])
    assert np.array_equal(result["numbers"], [6])


def test_reference_parser_separates_targets_from_predictions(tmp_path):
    result = arrays(tmp_path)
    assert result["target_energy"].tolist() == [-1.5]
    assert result["energy"].tolist() == [-1.4]
    assert np.array_equal(result["target_force"], [[1, 2, 3]])
    assert np.array_equal(result["force"], [[1.1, 2.1, 3.1]])


def test_reference_parser_preserves_geometry_pointers(tmp_path):
    result = arrays(tmp_path, FRAME + FRAME.replace("fixture:1", "fixture:2"))
    assert result["pointers"].tolist() == [0, 1, 2]
    assert result["identities"].tolist() == ["fixture:1", "fixture:2"]


def test_reference_parser_rejects_repeated_identities(tmp_path):
    with pytest.raises(ContractError, match="repeats"):
        arrays(tmp_path, FRAME + FRAME)


def test_reference_parser_rejects_empty_output(tmp_path):
    with pytest.raises(ContractError, match="empty"):
        arrays(tmp_path, "")


def test_reference_parser_rejects_nonfinite_energy(tmp_path):
    with pytest.raises(ContractError, match="nonfinite"):
        arrays(tmp_path, FRAME.replace("PRED_energy=-1.4", "PRED_energy=nan"))


def test_reference_parser_rejects_nonfinite_force(tmp_path):
    with pytest.raises(ContractError, match="nonfinite"):
        arrays(tmp_path, FRAME.replace("1.1 2.1 3.1", "inf 2.1 3.1"))


def test_reference_parser_rejects_missing_target_energy(tmp_path):
    with pytest.raises(ContractError, match="metadata"):
        arrays(tmp_path, FRAME.replace("REF_energy=-1.5 ", ""))


def test_reference_parser_rejects_wrong_force_axes(tmp_path):
    with pytest.raises(ContractError, match="malformed"):
        arrays(tmp_path, FRAME.replace("PRED_forces:R:3", "PRED_forces:R:2"))


def test_reference_parser_rejects_duplicate_properties(tmp_path):
    with pytest.raises(ContractError, match="duplicate"):
        arrays(tmp_path, FRAME.replace("PRED_forces:R:3", "REF_forces:R:3"))


def test_reference_parser_rejects_nonnumeric_force_type(tmp_path):
    with pytest.raises(ContractError, match="scalar type"):
        arrays(tmp_path, FRAME.replace("PRED_forces:R:3", "PRED_forces:S:3"))


def test_reference_parser_rejects_periodic_cell(tmp_path):
    with pytest.raises(ContractError, match="periodic"):
        arrays(tmp_path, FRAME.replace('pbc="F F F"', 'pbc="T F F"'))


def test_reference_parser_rejects_duplicate_energy_keys(tmp_path):
    with pytest.raises(ContractError, match="duplicate"):
        arrays(tmp_path, FRAME.replace("PRED_energy=-1.4", "PRED_energy=-1.4 PRED_energy=-1.3"))


def test_reference_parser_rejects_unsupported_element(tmp_path):
    with pytest.raises(ContractError, match="unsupported"):
        arrays(tmp_path, FRAME.replace("\nC ", "\nFe "))


def test_reference_parser_rejects_truncated_atom_row(tmp_path):
    with pytest.raises(ContractError, match="column count"):
        arrays(tmp_path, FRAME.rsplit("3.1", 1)[0])


def test_reference_parser_rejects_excess_atom_count(tmp_path):
    with pytest.raises(ResourceError, match="budget"):
        arrays(tmp_path, FRAME.replace("1\n", "513\n", 1))


def test_reference_parser_rejects_excess_frame_count(tmp_path):
    with pytest.raises(ResourceError, match="budget"):
        arrays(tmp_path, FRAME + FRAME.replace("fixture:1", "fixture:2"), maximum_frames=1)


def test_reference_parser_rejects_oversized_header(tmp_path):
    with pytest.raises(ResourceError, match="header budget"):
        arrays(tmp_path, "1\n" + ("a" * 65537) + "\n")
