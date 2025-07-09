# pyvoi/core/io.py

"""
Input/Output utilities for pyVOI.

This module provides functions for reading and writing data relevant to VOI analyses,
such as PSA samples and Net Benefit Arrays, primarily using CSV format for simplicity
and broad compatibility. Integration with more complex formats (e.g., HDF5, Parquet)
or database connectors could be added in the future if needed.
"""

import csv  # Using standard csv module for minimal dependencies
from typing import List, Optional

import numpy as np

from pyvoi.config import DEFAULT_DTYPE
from pyvoi.core.data_structures import NetBenefitArray, PSASample
from pyvoi.exceptions import FileFormatError, InputError


def _read_csv_data(filepath: str, delimiter: str) -> List[List[float]]:
    """Read numerical data from a CSV file."""
    data_rows = []
    try:
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row_idx, row in enumerate(reader):
                try:
                    data_rows.append([float(x) for x in row])
                except ValueError as e_val:
                    raise FileFormatError(
                        f"Error parsing CSV file '{filepath}' at row {row_idx + 1}: "
                        f"Non-numeric value found. Original error: {e_val}",
                    ) from e_val
    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileFormatError(
            f"Failed to read or parse CSV file '{filepath}': {e}"
        ) from e
    return data_rows


def read_net_benefit_array_csv(
    filepath: str,
    strategy_names: Optional[List[str]] = None,
    delimiter: str = ",",
    skip_header: bool = False,
    dtype: Optional[np.dtype] = None,
) -> NetBenefitArray:
    """Read a NetBenefitArray from a CSV file.

    The CSV file is expected to have samples as rows and strategies as columns.
    If `strategy_names` are not provided and `skip_header` is False, the first
    row is assumed to be the header containing strategy names.

    Args:
        filepath (str): Path to the CSV file.
        strategy_names (Optional[List[str]]): Optional list of strategy names.
            If provided, it overrides any header in the CSV.
        delimiter (str): Delimiter used in the CSV file. Defaults to ','.
        skip_header (bool): If True, skips the first row of the CSV file.
            Useful if strategy_names are provided separately or there's no header.
        dtype (np.dtype): NumPy data type to interpret the values. Defaults to DEFAULT_DTYPE.

    Returns
    -------
        NetBenefitArray: An instance of NetBenefitArray.

    Raises
    ------
        FileNotFoundError: If the filepath does not exist.
        FileFormatError: If the CSV file format is invalid or cannot be parsed.
        InputError: If provided strategy_names are inconsistent with CSV columns.
    """
    parsed_strategy_names = None
    if not skip_header and strategy_names is None:
        try:
            with open(filepath, "r", newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=delimiter)
                parsed_strategy_names = next(reader)
        except (FileNotFoundError, StopIteration):
            pass  # Handled by _read_csv_data

    data_rows = _read_csv_data(filepath, delimiter)
    if not data_rows:
        raise FileFormatError(f"CSV file '{filepath}' contains no data rows.")

    if not skip_header and parsed_strategy_names is not None:
        data_rows.pop(0)

    values = np.array(data_rows, dtype=dtype)

    # Determine final strategy names
    final_strategy_names = strategy_names
    if final_strategy_names is None and parsed_strategy_names is not None:
        final_strategy_names = [str(name).strip() for name in parsed_strategy_names]

    if (
        final_strategy_names is not None
        and len(final_strategy_names) != values.shape[1]
    ):
        raise InputError(
            f"Number of provided/parsed strategy_names ({len(final_strategy_names)}) "
            f"does not match number of columns in CSV data ({values.shape[1]}).",
        )

    return NetBenefitArray(values=values, strategy_names=final_strategy_names)


def write_net_benefit_array_csv(
    nba: NetBenefitArray,
    filepath: str,
    delimiter: str = ",",
    write_header: bool = True,
) -> None:
    """Write a NetBenefitArray to a CSV file.

    Args:
        nba (NetBenefitArray): The NetBenefitArray instance to write.
        filepath (str): Path to the output CSV file.
        delimiter (str): Delimiter to use in the CSV file. Defaults to ','.
        write_header (bool): If True and strategy_names are available in nba,
                             writes them as the first row. Defaults to True.

    Raises
    ------
        IOError: If writing to the file fails.
    """
    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)

            if write_header and nba.strategy_names is not None:
                writer.writerow(nba.strategy_names)

            for row in nba.values:
                writer.writerow(row)
    except IOError as e:
        raise IOError(
            f"Failed to write NetBenefitArray to CSV file '{filepath}': {e}"
        ) from e


def read_psa_samples_csv(
    filepath: str,
    parameter_names: Optional[List[str]] = None,
    delimiter: str = ",",
    skip_header: bool = False,
    dtype: Optional[np.dtype] = None,
) -> PSASample:
    """Read PSA samples from a CSV file into a PSASample object.

    The CSV file is expected to have samples as rows and parameters as columns.
    If `parameter_names` are not provided and `skip_header` is False, the first
    row is assumed to be the header containing parameter names.

    Args:
        filepath (str): Path to the CSV file.
        parameter_names (Optional[List[str]]): Optional list of parameter names.
            If provided, it overrides any header in the CSV.
        delimiter (str): Delimiter used in the CSV file. Defaults to ','.
        skip_header (bool): If True, skips the first row of the CSV file.
        dtype (np.dtype): NumPy data type for the parameter values. Defaults to DEFAULT_DTYPE.

    Returns
    -------
        PSASample: An instance of PSASample.

    Raises
    ------
        FileNotFoundError: If the filepath does not exist.
        FileFormatError: If the CSV file format is invalid.
        InputError: If provided parameter_names are inconsistent with CSV columns.
    """
    parsed_param_names = None
    if not skip_header and parameter_names is None:
        try:
            with open(filepath, "r", newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=delimiter)
                parsed_param_names = next(reader)
        except (FileNotFoundError, StopIteration):
            pass  # Handled by _read_csv_data

    data_rows = _read_csv_data(filepath, delimiter)
    if not data_rows:
        raise FileFormatError(f"CSV file '{filepath}' contains no data rows.")

    if not skip_header and parsed_param_names is not None:
        data_rows.pop(0)

    data_array = np.array(data_rows, dtype=dtype)  # (n_samples, n_parameters)

    # Determine final parameter names
    final_param_names = parameter_names
    if final_param_names is None and parsed_param_names is not None:
        final_param_names = [str(name).strip() for name in parsed_param_names]

    if final_param_names is None:
        # Auto-generate parameter names if none are available
        final_param_names = [f"param_{i}" for i in range(data_array.shape[1])]
        # Or raise error: raise InputError("Parameter names must be provided or present in CSV header.")
    elif len(final_param_names) != data_array.shape[1]:
        raise InputError(
            f"Number of provided/parsed parameter_names ({len(final_param_names)}) "
            f"does not match number of columns in CSV data ({data_array.shape[1]}).",
        )

    parameters_dict = {
        name: data_array[:, i] for i, name in enumerate(final_param_names)
    }

    return PSASample(parameters=parameters_dict)


def write_psa_samples_csv(
    psa_sample: PSASample,
    filepath: str,
    delimiter: str = ",",
    write_header: bool = True,
) -> None:
    """Write a PSASample object to a CSV file.

    Args:
        psa_sample (PSASample): The PSASample instance to write.
        filepath (str): Path to the output CSV file.
        delimiter (str): Delimiter to use in the CSV file. Defaults to ','.
        write_header (bool): If True, writes parameter names as the first row.
                             Requires PSASample.parameters to be a dictionary. Defaults to True.

    Raises
    ------
        InputError: If psa_sample.parameters is not a dictionary and write_header is True.
        IOError: If writing to the file fails.
    """
    if not isinstance(psa_sample.parameters, dict):
        if write_header:
            raise InputError(
                "Cannot write header for PSASample if 'parameters' attribute is not a dictionary.",
            )
        # Handle non-dict PSASample (e.g., xarray) if supported, or raise error
        # For now, assume if not dict, it's not directly writable by this simple CSV function
        # without more complex handling.
        raise InputError(
            "This CSV writer currently only supports PSASample with dictionary-based parameters.",
        )

    param_names = list(psa_sample.parameters.keys())
    if not param_names:
        # Handle empty PSASample (write empty file or raise error)
        try:
            with open(filepath, "w", newline="") as csvfile:  # Creates an empty file
                if write_header:  # Write empty header if requested
                    writer = csv.writer(csvfile, delimiter=delimiter)
                    writer.writerow([])
            return
        except IOError as e:
            raise IOError(
                f"Failed to write empty PSASample to CSV file '{filepath}': {e}"
            ) from e

    # Assuming all parameter arrays have the same length (n_samples), validated by PSASample
    n_samples = psa_sample.n_samples

    try:
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)

            if write_header:
                writer.writerow(param_names)

            for i in range(n_samples):
                row = [psa_sample.parameters[name][i] for name in param_names]
                writer.writerow(row)
    except IOError as e:
        raise IOError(f"Failed to write PSASample to CSV file '{filepath}': {e}") from e
    except KeyError as e:
        # Should ideally be caught by PSASample validation if param_names derived from keys
        raise FileFormatError(
            f"Inconsistency in PSASample data during CSV write: {e}"
        ) from e


if __name__ == "__main__":
    # Create dummy data for testing
    print("--- Testing IO functions ---")

    # Test NetBenefitArray I/O
    nba_values = np.array(
        [[100, 110.5], [90, 120.0], [105, 115.33]], dtype=DEFAULT_DTYPE
    )
    nba_strategies = ["Strategy X", "Strategy Y"]
    original_nba = NetBenefitArray(values=nba_values, strategy_names=nba_strategies)

    nba_filepath = "temp_nba_test.csv"
    try:
        print(f"Writing NBA to {nba_filepath}...")
        write_net_benefit_array_csv(original_nba, nba_filepath)
        print("Reading NBA back...")
        read_nba = read_net_benefit_array_csv(
            nba_filepath, dtype=None
        )  # strategy_names read from header

        if not np.array_equal(original_nba.values, read_nba.values):
            raise ValueError("NBA values differ after I/O.")
        if not (original_nba.strategy_names == read_nba.strategy_names):
            raise ValueError("NBA strategy names differ after I/O.")
        print("NetBenefitArray I/O test PASSED.")

        # Test reading with explicit strategy names
        read_nba_explicit_names = read_net_benefit_array_csv(
            nba_filepath,
            strategy_names=["Custom X", "Custom Y"],
            skip_header=True,  # Important if providing names and CSV has header
        )
        if not np.array_equal(original_nba.values, read_nba_explicit_names.values):
            raise ValueError("NBA values differ after I/O with explicit names.")
        if not (read_nba_explicit_names.strategy_names == ["Custom X", "Custom Y"]):
            raise ValueError("NBA strategy names differ after I/O with explicit names.")
        print("NetBenefitArray I/O with explicit names and skip_header PASSED.")

    except Exception as e:
        print(f"NetBenefitArray I/O test FAILED: {e}")
    finally:
        import os

        if os.path.exists(nba_filepath):
            os.remove(nba_filepath)

    # Test PSASample I/O
    psa_params = {
        "cost_A": np.array([10, 12, 11], dtype=DEFAULT_DTYPE),
        "effect_A": np.array([0.8, 0.85, 0.82], dtype=DEFAULT_DTYPE),
        "utility_B": np.array([0.9, 0.88, 0.91], dtype=DEFAULT_DTYPE),
    }
    original_psa = PSASample(parameters=psa_params)

    psa_filepath = "temp_psa_test.csv"
    try:
        print(f"\nWriting PSA to {psa_filepath}...")
        write_psa_samples_csv(original_psa, psa_filepath)
        print("Reading PSA back...")
        read_psa = read_psa_samples_csv(
            psa_filepath, dtype=None
        )  # param_names read from header

        if not (original_psa.n_samples == read_psa.n_samples):
            raise ValueError("PSA n_samples differ.")
        if not (set(original_psa.parameter_names) == set(read_psa.parameter_names)):
            raise ValueError("PSA parameter names differ.")
        for name in original_psa.parameter_names:
            if not np.array_equal(
                original_psa.parameters[name], read_psa.parameters[name]
            ):
                raise ValueError(f"PSA values for parameter '{name}' differ after I/O.")
        print("PSASample I/O test PASSED.")

        # Test reading with explicit parameter names
        read_psa_explicit_names = read_psa_samples_csv(
            psa_filepath,
            parameter_names=["p1", "p2", "p3"],
            skip_header=True,
        )
        if not (original_psa.n_samples == read_psa_explicit_names.n_samples):
            raise ValueError("PSA n_samples differ with explicit names.")
        if not (read_psa_explicit_names.parameter_names == ["p1", "p2", "p3"]):
            raise ValueError("PSA parameter names differ with explicit names.")
        # Values should still match column-wise
        if not np.array_equal(
            original_psa.parameters["cost_A"], read_psa_explicit_names.parameters["p1"]
        ):
            raise ValueError("PSA values differ with explicit names.")
        print("PSASample I/O with explicit names and skip_header PASSED.")

    except Exception as e:
        print(f"PSASample I/O test FAILED: {e}")
    finally:
        import os

        if os.path.exists(psa_filepath):
            os.remove(psa_filepath)

    print("\n--- IO Testing Done ---")
