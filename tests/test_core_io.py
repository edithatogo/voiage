"""Tests for `voiage.core.io` CSV helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from voiage.core.io import (
    FileFormatError,
    import_callable,
    read_parameter_set_csv,
    read_value_array_csv,
    write_parameter_set_csv,
    write_value_array_csv,
)
from voiage.schema import ParameterSet, ValueArray

if TYPE_CHECKING:
    from pathlib import Path


def _value_array() -> ValueArray:
    dataset = xr.Dataset(
        {
            "net_benefit": (
                ("n_samples", "n_strategies"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        coords={
            "n_samples": np.array([0, 1]),
            "n_strategies": np.array([0, 1]),
            "strategy": ("n_strategies", ["A", "B"]),
        },
    )
    return ValueArray(dataset=dataset)


def _parameter_set() -> ParameterSet:
    dataset = xr.Dataset(
        {
            "cost": ("n_samples", np.array([10.0, 20.0, 30.0])),
            "effect": ("n_samples", np.array([0.1, 0.2, 0.3])),
        },
        coords={"n_samples": np.array([0, 1, 2])},
    )
    return ParameterSet(dataset=dataset)


def test_value_array_csv_round_trip(tmp_path: Path) -> None:
    filepath = tmp_path / "value_array.csv"
    value_array = _value_array()

    write_value_array_csv(value_array, str(filepath))

    restored = read_value_array_csv(
        str(filepath), option_names=value_array.strategy_names, skip_header=True
    )

    np.testing.assert_array_equal(restored.numpy_values, value_array.numpy_values)
    assert restored.strategy_names == ["A", "B"]


def test_value_array_csv_skip_header(tmp_path: Path) -> None:
    filepath = tmp_path / "value_array_with_header.csv"
    filepath.write_text("A,B\n1,2\n3,4\n")

    restored = read_value_array_csv(str(filepath), skip_header=True)

    assert restored.numpy_values.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert restored.strategy_names == ["Option 1", "Option 2"]


def test_value_array_csv_single_row_round_trip(tmp_path: Path) -> None:
    filepath = tmp_path / "single_row_value_array.csv"
    filepath.write_text("1,2\n")

    restored = read_value_array_csv(str(filepath), option_names=["A", "B"])

    assert restored.numpy_values.shape == (1, 2)
    assert restored.numpy_values.tolist() == [[1.0, 2.0]]
    assert restored.strategy_names == ["A", "B"]


def test_value_array_csv_single_value_round_trip(tmp_path: Path) -> None:
    filepath = tmp_path / "single_value_value_array.csv"
    filepath.write_text("1\n")

    restored = read_value_array_csv(str(filepath), option_names=["A"])

    assert restored.numpy_values.shape == (1, 1)
    assert restored.numpy_values.tolist() == [[1.0]]


def test_value_array_csv_option_name_mismatch(tmp_path: Path) -> None:
    filepath = tmp_path / "value_array.csv"
    filepath.write_text("1,2\n3,4\n")

    with pytest.raises(FileFormatError, match="strategies"):
        read_value_array_csv(str(filepath), option_names=["A"])


def test_value_array_csv_rejects_empty_option_names(tmp_path: Path) -> None:
    filepath = tmp_path / "value_array.csv"
    filepath.write_text("1,2\n3,4\n")

    with pytest.raises(FileFormatError, match="strategies"):
        read_value_array_csv(str(filepath), option_names=[])


def test_value_array_csv_normalizes_empty_body_with_explicit_names(
    tmp_path: Path,
) -> None:
    filepath = tmp_path / "value_array_empty.csv"
    filepath.write_text("A,B\n\n")

    restored = read_value_array_csv(
        str(filepath), option_names=[1, 2], skip_header=True
    )

    assert restored.numpy_values.shape == (0, 2)
    assert restored.strategy_names == ["1", "2"]


def test_parameter_set_csv_round_trip(tmp_path: Path) -> None:
    filepath = tmp_path / "parameter_set.csv"
    parameter_set = _parameter_set()

    write_parameter_set_csv(parameter_set, str(filepath))

    restored = read_parameter_set_csv(
        str(filepath), parameter_names=parameter_set.parameter_names, skip_header=True
    )

    assert restored.n_samples == parameter_set.n_samples
    np.testing.assert_array_equal(
        restored.parameters["cost"], parameter_set.parameters["cost"]
    )
    np.testing.assert_array_equal(
        restored.parameters["effect"], parameter_set.parameters["effect"]
    )
    assert restored.parameter_names == ["cost", "effect"]


def test_parameter_set_csv_skip_header(tmp_path: Path) -> None:
    filepath = tmp_path / "parameter_set_with_header.csv"
    filepath.write_text("cost,effect\n10,0.1\n20,0.2\n")

    restored = read_parameter_set_csv(str(filepath), skip_header=True)

    assert restored.parameter_names == ["param_1", "param_2"]
    assert restored.parameters["param_1"].tolist() == [10.0, 20.0]


def test_parameter_set_csv_parameter_name_mismatch(tmp_path: Path) -> None:
    filepath = tmp_path / "parameter_set.csv"
    filepath.write_text("10,0.1\n20,0.2\n")

    with pytest.raises(FileFormatError, match="parameters"):
        read_parameter_set_csv(str(filepath), parameter_names=["cost"])


def test_parameter_set_csv_rejects_empty_parameter_names(tmp_path: Path) -> None:
    filepath = tmp_path / "parameter_set.csv"
    filepath.write_text("10,0.1\n20,0.2\n")

    with pytest.raises(FileFormatError, match="parameters"):
        read_parameter_set_csv(str(filepath), parameter_names=[])


def test_parameter_set_csv_normalizes_empty_body_with_explicit_names(
    tmp_path: Path,
) -> None:
    filepath = tmp_path / "parameter_set_empty.csv"
    filepath.write_text("cost,effect\n\n")

    restored = read_parameter_set_csv(
        str(filepath), parameter_names=[1, 2], skip_header=True
    )

    assert restored.n_samples == 0
    assert restored.parameter_names == ["1", "2"]


def test_import_callable_resolves_builtin_callables() -> None:
    assert import_callable("builtins.len") is len


def test_import_callable_rejects_invalid_target_format() -> None:
    with pytest.raises(FileFormatError, match="module and attribute name"):
        import_callable("len")


def test_import_callable_rejects_missing_or_non_callable_targets() -> None:
    with pytest.raises(
        FileFormatError,
        match=r"module 'builtins' does not define 'missing_name'",
    ):
        import_callable("builtins.missing_name")

    with pytest.raises(
        FileFormatError,
        match=r"'builtins.__name__' does not resolve to a callable object",
    ):
        import_callable("builtins.__name__")

    with pytest.raises(FileFormatError, match="could not import module"):
        import_callable("does_not_exist.callable")


def test_read_value_array_csv_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileFormatError, match="Failed to read ValueArray"):
        read_value_array_csv(str(missing))


def test_read_parameter_set_csv_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileFormatError, match="Failed to read ParameterSet"):
        read_parameter_set_csv(str(missing))


def test_write_value_array_csv_raises_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filepath = tmp_path / "value_array.csv"

    def _raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raise_os_error)

    with pytest.raises(OSError, match="Failed to write ValueArray"):
        write_value_array_csv(_value_array(), str(filepath))


def test_write_value_array_csv_can_skip_header(tmp_path: Path) -> None:
    filepath = tmp_path / "value_array_no_header.csv"

    write_value_array_csv(_value_array(), str(filepath), write_header=False)

    assert filepath.read_text().splitlines() == ["1.0,2.0", "3.0,4.0"]


def test_write_parameter_set_csv_raises_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filepath = tmp_path / "parameter_set.csv"

    def _raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _raise_os_error)

    with pytest.raises(OSError, match="Failed to write ParameterSet"):
        write_parameter_set_csv(_parameter_set(), str(filepath))


def test_write_parameter_set_csv_can_skip_header(tmp_path: Path) -> None:
    filepath = tmp_path / "parameter_set_no_header.csv"

    write_parameter_set_csv(_parameter_set(), str(filepath), write_header=False)

    assert filepath.read_text().splitlines() == [
        "10.0,0.1",
        "20.0,0.2",
        "30.0,0.3",
    ]
