"""Input/Output utilities for voiage."""

from collections.abc import Callable, Sequence
import csv
import importlib
from typing import Any

import numpy as np
import xarray as xr

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import InputError
from voiage.schema import ParameterSet, ValueArray

_OPTION_NAME_COUNT_MISMATCH = (
    "Number of option_names does not match number of strategies."
)
_PARAMETER_NAME_COUNT_MISMATCH = (
    "Number of parameter_names does not match number of parameters."
)


def _read_value_array_error(filepath: str, exc: Exception) -> "FileFormatError":
    return FileFormatError(
        f"Failed to read ValueArray from CSV file '{filepath}': {exc}"
    )


def _write_value_array_error(filepath: str, exc: Exception) -> OSError:
    return OSError(f"Failed to write ValueArray to CSV file '{filepath}': {exc}")


def _read_parameter_set_error(filepath: str, exc: Exception) -> "FileFormatError":
    return FileFormatError(
        f"Failed to read ParameterSet from CSV file '{filepath}': {exc}"
    )


def _write_parameter_set_error(filepath: str, exc: Exception) -> OSError:
    return OSError(f"Failed to write ParameterSet to CSV file '{filepath}': {exc}")


def _read_csv_values(
    filepath: str,
    delimiter: str,
    skip_header: bool,
    dtype: object,
) -> np.ndarray:
    """Read CSV rows into a normalized 2D NumPy array."""
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader, None)

        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        return np.empty((0, 0), dtype=dtype)

    values = np.asarray([list(map(dtype, row)) for row in rows], dtype=dtype)
    if values.ndim == 1:  # pragma: no cover - CSV rows are always nested
        values = values.reshape(1, -1)
    return values


def _normalize_csv_names(
    names: Sequence[object] | None,
    count: int,
    default_prefix: str,
    mismatch_message: str,
) -> list[str]:
    """Normalize CSV labels to strings and validate their length."""
    if names is None:
        return [f"{default_prefix}{i + 1}" for i in range(count)]

    normalized_names = [str(name) for name in names]
    if len(normalized_names) != count:
        raise FileFormatError(mismatch_message)
    return normalized_names


class FileFormatError(InputError):
    """Raised when a file's format is invalid or inconsistent with expectations."""

    pass


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a dotted module path."""
    if "." not in path:
        raise FileFormatError("callable path must include a module and attribute name")

    module_name, attribute_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise FileFormatError(f"could not import module '{module_name}'") from exc

    try:
        attribute = getattr(module, attribute_name)
    except AttributeError as exc:
        raise FileFormatError(
            f"module '{module_name}' does not define '{attribute_name}'"
        ) from exc

    if not callable(attribute):
        raise FileFormatError(f"'{path}' does not resolve to a callable object")

    return attribute


def read_value_array_csv(
    filepath: str,
    option_names: Sequence[object] | None = None,
    delimiter: str = ",",
    skip_header: bool = False,
    dtype: object = DEFAULT_DTYPE,
) -> ValueArray:
    """Read a ValueArray from a CSV file."""
    try:
        values = _read_csv_values(filepath, delimiter, skip_header, dtype)
        expected_option_count = len(option_names) if option_names is not None else None
        if expected_option_count is not None and values.shape[1] == 0:
            values = np.empty((0, expected_option_count), dtype=dtype)

        final_option_names = _normalize_csv_names(
            option_names,
            values.shape[1],
            "Option ",
            _OPTION_NAME_COUNT_MISMATCH,
        )

        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(values.shape[0]),
                "n_strategies": np.arange(values.shape[1]),
                "strategy": ("n_strategies", final_option_names),
            },
        )
        return ValueArray(dataset=dataset)

    except (OSError, ValueError) as e:
        raise _read_value_array_error(filepath, e) from e


def write_value_array_csv(
    value_array: ValueArray,
    filepath: str,
    delimiter: str = ",",
    write_header: bool = True,
) -> None:
    """Write a ValueArray to a CSV file."""
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            if write_header:
                writer.writerow(value_array.strategy_names)
            writer.writerows(value_array.numpy_values.tolist())
    except OSError as e:
        raise _write_value_array_error(filepath, e) from e


def read_parameter_set_csv(
    filepath: str,
    parameter_names: Sequence[object] | None = None,
    delimiter: str = ",",
    skip_header: bool = False,
    dtype: object = DEFAULT_DTYPE,
) -> ParameterSet:
    """Read PSA samples from a CSV file into a ParameterSet object."""
    try:
        values = _read_csv_values(filepath, delimiter, skip_header, dtype)
        expected_parameter_count = (
            len(parameter_names) if parameter_names is not None else None
        )
        if expected_parameter_count is not None and values.shape[1] == 0:
            values = np.empty((0, expected_parameter_count), dtype=dtype)

        final_parameter_names = _normalize_csv_names(
            parameter_names,
            values.shape[1],
            "param_",
            _PARAMETER_NAME_COUNT_MISMATCH,
        )

        dataset = xr.Dataset(
            {
                name: (("n_samples",), values[:, i])
                for i, name in enumerate(final_parameter_names)
            },
            coords={"n_samples": np.arange(values.shape[0])},
        )
        return ParameterSet(dataset=dataset)

    except (OSError, ValueError) as e:
        raise _read_parameter_set_error(filepath, e) from e


def write_parameter_set_csv(
    parameter_set: ParameterSet,
    filepath: str,
    delimiter: str = ",",
    write_header: bool = True,
) -> None:
    """Write a ParameterSet object to a CSV file."""
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            if write_header:
                writer.writerow(parameter_set.parameter_names)

            param_values = np.vstack(list(parameter_set.parameters.values())).T
            writer.writerows(param_values.tolist())

    except (OSError, ValueError) as e:
        raise _write_parameter_set_error(filepath, e) from e
