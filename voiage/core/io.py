"""Input/Output utilities for voiage."""

import csv
from typing import Any, List, Optional

import numpy as np
import xarray as xr

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import InputError
from voiage.schema import ParameterSet, ValueArray


class FileFormatError(InputError):
    """Raised when a file's format is invalid or inconsistent with expectations."""

    pass


def read_value_array_csv(
    filepath: str,
    option_names: Optional[List[str]] = None,
    delimiter: str = ",",
    skip_header: bool = False,
    dtype: Any = DEFAULT_DTYPE,
) -> ValueArray:
    """Read a ValueArray from a CSV file."""
    try:
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            if skip_header:
                next(reader)

            data = [list(map(dtype, row)) for row in reader]
            values = np.array(data, dtype=dtype)

        if option_names and len(option_names) != values.shape[1]:
            raise FileFormatError(
                "Number of option_names does not match number of columns."
            )

        final_option_names = option_names or [
            f"Option {i+1}" for i in range(values.shape[1])
        ]

        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(values.shape[0]),
                "n_strategies": np.arange(values.shape[1]),
                "strategy": ("n_strategies", final_option_names),
            },
        )
        return ValueArray(dataset=dataset)

    except (IOError, ValueError) as e:
        raise FileFormatError(
            f"Failed to read ValueArray from CSV file '{filepath}': {e}"
        ) from e


def write_value_array_csv(
    value_array: ValueArray,
    filepath: str,
    delimiter: str = ",",
    write_header: bool = True,
):
    """Write a ValueArray to a CSV file."""
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            if write_header:
                writer.writerow(value_array.option_names)
            writer.writerows(value_array.values.tolist())
    except IOError as e:
        raise IOError(
            f"Failed to write ValueArray to CSV file '{filepath}': {e}"
        ) from e


def read_parameter_set_csv(
    filepath: str,
    parameter_names: Optional[List[str]] = None,
    delimiter: str = ",",
    skip_header: bool = False,
    dtype: Any = DEFAULT_DTYPE,
) -> ParameterSet:
    """Read PSA samples from a CSV file into a ParameterSet object."""
    try:
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            if skip_header:
                next(reader)

            data = [list(map(dtype, row)) for row in reader]
            values = np.array(data, dtype=dtype)

        if parameter_names and len(parameter_names) != values.shape[1]:
            raise FileFormatError(
                "Number of parameter_names does not match number of columns."
            )

        final_parameter_names = parameter_names or [
            f"param_{i+1}" for i in range(values.shape[1])
        ]

        param_dict = {
            name: values[:, i] for i, name in enumerate(final_parameter_names)
        }

        dataset = xr.Dataset(
            {k: (("n_samples",), v) for k, v in param_dict.items()},
            coords={"n_samples": np.arange(values.shape[0])},
        )
        return ParameterSet(dataset=dataset)

    except (IOError, ValueError) as e:
        raise FileFormatError(
            f"Failed to read ParameterSet from CSV file '{filepath}': {e}"
        ) from e


def write_parameter_set_csv(
    parameter_set: ParameterSet,
    filepath: str,
    delimiter: str = ",",
    write_header: bool = True,
):
    """Write a ParameterSet object to a CSV file."""
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            if write_header:
                writer.writerow(parameter_set.parameter_names)

            param_values = np.vstack(list(parameter_set.parameters.values())).T
            writer.writerows(param_values.tolist())

    except (IOError, ValueError) as e:
        raise IOError(
            f"Failed to write ParameterSet to CSV file '{filepath}': {e}"
        ) from e
