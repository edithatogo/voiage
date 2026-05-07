# voiage/core/utils.py

"""Utility functions for the voiage library."""

from collections.abc import Sequence
from typing import cast

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import DimensionMismatchError, InputError
from voiage.schema import ValueArray

_INPUT_ARRAY_TYPE_MESSAGE = "must be a NumPy array."
_EMPTY_ARRAY_MESSAGE = "cannot be empty."
_WTP_TYPE_MESSAGE = "WTP must be a float, int, or NumPy array."
_WTP_DIMENSION_MESSAGE = "WTP array cannot have more than 2 dimensions."
_NB_ARRAY_MESSAGE = "nb_array must be a NumPy array or NetBenefitArray instance."


def _input_array_type_error(name: str, actual_type: type[object]) -> InputError:
    return InputError(f"{name} {_INPUT_ARRAY_TYPE_MESSAGE} Got {actual_type}.")


def _array_dimension_error(
    name: str, expected_ndim_tuple: tuple[int, ...], actual_ndim: int
) -> DimensionMismatchError:
    return DimensionMismatchError(
        f"{name} must have {expected_ndim_tuple} dimension(s). Got {actual_ndim}."
    )


def _empty_array_error(name: str) -> InputError:
    return InputError(f"{name} {_EMPTY_ARRAY_MESSAGE}")


def _dtype_error(
    name: str, dtype_to_check: np.dtype | type | str, actual: object
) -> InputError:
    return InputError(f"{name} must have dtype {dtype_to_check}. Got {actual}.")


def _shape_tuple_error(
    expected_shape: Sequence[int | None], actual_ndim: int
) -> DimensionMismatchError:
    return DimensionMismatchError(
        f"Expected shape tuple length {len(expected_shape)} does not match "
        f"array ndim {actual_ndim}."
    )


def _shape_dimension_error(
    name: str,
    index: int,
    actual_dim_size: int,
    expected_dim_size: int,
    actual_shape: tuple[int, ...],
    expected_shape: Sequence[int | None],
) -> DimensionMismatchError:
    return DimensionMismatchError(
        f"Dimension {index} of {name} has size {actual_dim_size}, "
        f"but expected {expected_dim_size}. Shape: {actual_shape}, Expected: {expected_shape}"
    )


def _costs_effects_shape_error(
    costs_shape: tuple[int, ...], effects_shape: tuple[int, ...]
) -> DimensionMismatchError:
    return DimensionMismatchError(
        f"Costs shape {costs_shape} and effects shape {effects_shape} must match."
    )


def _wtp_type_error() -> InputError:
    return InputError(_WTP_TYPE_MESSAGE)


def _wtp_dimension_error() -> DimensionMismatchError:
    return DimensionMismatchError(_WTP_DIMENSION_MESSAGE)


def _nb_array_type_error() -> InputError:
    return InputError(_NB_ARRAY_MESSAGE)


def check_input_array(
    arr: np.ndarray,
    expected_ndim: int | Sequence[int],
    name: str = "Input array",
    expected_dtype: np.dtype | type | str | None = None,
    allow_empty: bool = False,
    expected_shape: Sequence[int | None] | None = None,
) -> None:
    """Validate a NumPy array against expected dimensions, dtype, and shape.

    Args:
        arr (np.ndarray): The array to validate.
        expected_ndim (Union[int, Sequence[int]]): The expected number of dimensions,
            or a sequence of allowed numbers of dimensions.
        name (str): Name of the array for error messages. Defaults to "Input array".
        expected_dtype (Optional[np.dtype]): Expected data type. If None, uses DEFAULT_DTYPE.
                                             If "any", skips dtype check.
        allow_empty (bool): If False, raises error if the array is empty (size 0).
        expected_shape (Optional[Sequence[Optional[int]]]): Expected shape.
            Use None for dimensions that can have any size (e.g., (None, 3) for N-by-3).

    Raises
    ------
        InputError: If arr is not a NumPy array or fails validation.
        DimensionMismatchError: If dimensions or shape are incorrect.
    """
    if not isinstance(arr, np.ndarray):
        raise _input_array_type_error(name, type(arr))

    if isinstance(expected_ndim, int):
        expected_ndim_tuple: tuple[int, ...] = (expected_ndim,)
    else:
        expected_ndim_tuple = tuple(expected_ndim)

    if arr.ndim not in expected_ndim_tuple:
        raise _array_dimension_error(name, expected_ndim_tuple, arr.ndim)

    if not allow_empty and arr.size == 0:
        raise _empty_array_error(name)

    dtype_to_check = expected_dtype if expected_dtype is not None else DEFAULT_DTYPE
    if isinstance(dtype_to_check, str) and dtype_to_check.lower() == "any":
        pass  # Skip dtype check
    elif arr.dtype != dtype_to_check:
        # Allow flexibility for JAX arrays: accept both float32 and float64 for numerical arrays
        if (
            hasattr(arr, "dtype")
            and arr.dtype in [np.float32, np.float64]
            and dtype_to_check in [np.float32, np.float64]
        ):
            pass  # Allow both float32 and float64 for numerical compatibility
        else:
            raise _dtype_error(name, dtype_to_check, arr.dtype)

    if expected_shape is not None:
        if len(expected_shape) != arr.ndim:
            raise _shape_tuple_error(expected_shape, arr.ndim)
        for i, (expected_dim_size, actual_dim_size) in enumerate(
            zip(expected_shape, arr.shape, strict=False)
        ):
            if expected_dim_size is not None and expected_dim_size != actual_dim_size:
                raise _shape_dimension_error(
                    name,
                    i,
                    actual_dim_size,
                    expected_dim_size,
                    arr.shape,
                    expected_shape,
                )


def calculate_net_benefit(
    costs: np.ndarray, effects: np.ndarray, wtp: float | np.ndarray
) -> np.ndarray:
    """Calculate net monetary benefit (NMB).

    NMB = (effects * wtp) - costs.
    Supports scalar or array WTP for threshold analysis.

    Args:
        costs (np.ndarray): Array of costs. Shape (n_samples, n_strategies) or (n_samples,).
        effects (np.ndarray): Array of effects (e.g., QALYs). Shape (n_samples, n_strategies) or (n_samples,).
                              Must match the shape of `costs`.
        wtp (Union[float, np.ndarray]): Willingness-to-pay threshold.
            - If float: scalar WTP applied to all.
            - If np.ndarray: WTP values, e.g., for different thresholds or time points.
                             Broadcasting rules apply. If WTP is an array, its shape
                             should be compatible for broadcasting with effects and costs.
                             E.g., (n_wtp_thresholds,) or (n_samples, n_wtp_thresholds).

    Returns
    -------
        np.ndarray: Array of net monetary benefits. Shape will depend on broadcasting with WTP.
                    If WTP is scalar, shape matches costs/effects.
                    If WTP is (k,), NMB shape will be (n_samples, n_strategies, k) or (n_samples, k).

    Raises
    ------
        DimensionMismatchError: If shapes of costs and effects are incompatible.
        InputError: If inputs are not valid NumPy arrays or WTP is invalid.
    """
    check_input_array(
        costs,
        expected_ndim=[1, 2],
        name="costs",
        expected_dtype="any",
        allow_empty=False,
    )
    check_input_array(
        effects,
        expected_ndim=[1, 2],
        name="effects",
        expected_dtype="any",
        allow_empty=False,
    )

    if costs.shape != effects.shape:
        raise _costs_effects_shape_error(costs.shape, effects.shape)

    if not isinstance(wtp, (float, int, np.ndarray)):
        raise _wtp_type_error()

    wtp_arr = np.asarray(wtp, dtype=DEFAULT_DTYPE)
    if (
        wtp_arr.ndim > 2
    ):  # Allowing 0D (scalar), 1D (thresholds), 2D (e.g. samples x thresholds)
        raise _wtp_dimension_error()
    if np.any(wtp_arr < 0):
        # Depending on context, WTP could be negative, but usually non-negative.
        # print("Warning: WTP contains negative values.")
        pass

    # NMB calculation: (effects * wtp) - costs
    # NumPy's broadcasting rules will handle scalar or array WTP.
    # If costs/effects are (N, S) and WTP is (K,), result might be (N, S, K)
    # If WTP is scalar, result is (N, S)
    # To ensure consistent broadcasting, we might need to reshape WTP if it's 1D (K,)
    # to be (1, 1, K) or (K, 1, 1) depending on desired output structure.
    # For now, let's assume standard broadcasting is sufficient.
    # If effects is (N,S) and wtp is (K), (effects_reshaped_for_wtp * wtp_reshaped)
    # effects_expanded = effects[..., np.newaxis] if wtp_arr.ndim > 0 and costs.ndim > wtp_arr.ndim else effects
    # costs_expanded = costs[..., np.newaxis] if wtp_arr.ndim > 0 and costs.ndim > wtp_arr.ndim else costs

    # A common way to handle (N,S) and (K,) WTP to get (N,S,K) is:
    # nmb = effects[:, :, np.newaxis] * wtp_arr[np.newaxis, np.newaxis, :] - costs[:, :, np.newaxis]
    # This can get complex. Let's simplify for common cases.

    if wtp_arr.ndim == 0:  # Scalar WTP
        nmb = (effects * wtp_arr) - costs
    elif (
        wtp_arr.ndim == 1 and costs.ndim == 2
    ):  # WTP is (K,), costs/effects are (N,S) -> NMB (N,S,K)
        nmb = (effects[..., np.newaxis] * wtp_arr) - costs[..., np.newaxis]
    elif (
        wtp_arr.ndim == 1 and costs.ndim == 1
    ):  # WTP is (K,), costs/effects are (N,) -> NMB (N,K)
        nmb = (effects[:, np.newaxis] * wtp_arr) - costs[:, np.newaxis]
    elif (
        wtp_arr.ndim == 2 and costs.ndim == 2
    ):  # WTP (N,K), costs/effects (N,S) -> NMB (N,S,K) (if S=1 for WTP) or complex
        # This case requires careful thought on shapes. If WTP is (N_samples, K_thresholds)
        # and costs/effects are (N_samples, S_strategies),
        # nmb = (effects[:, :, np.newaxis] * wtp_arr[:, np.newaxis, :]) - costs[:, :, np.newaxis]
        # This would result in (N, S, K)
        # For now, let's assume if wtp_arr is 2D, it's (N_samples, 1) or compatible with (N_samples, S_strategies) directly
        nmb = (effects * wtp_arr) - costs  # Relies on standard broadcasting
    else:  # Fallback to standard broadcasting, which might be what's desired or raise error
        nmb = (effects * wtp_arr) - costs

    return cast("np.ndarray", nmb.astype(DEFAULT_DTYPE, copy=False))


def get_optimal_strategy_index(
    nb_array: np.ndarray | ValueArray,
) -> np.ndarray:
    """Determine the optimal strategy for each PSA sample.

    The optimal strategy for a given sample is the one with the maximum
    net benefit.

    Args:
        nb_array (Union[np.ndarray, NetBenefitArray]): A 2D NumPy array of net benefits
            (samples x strategies) or a NetBenefitArray instance.

    Returns
    -------
        np.ndarray: A 1D array of integers, where each element is the index of the
                    optimal strategy for the corresponding sample.
    """
    if isinstance(nb_array, ValueArray):
        values: np.ndarray = nb_array.numpy_values
    elif isinstance(nb_array, np.ndarray):
        values = nb_array
    else:
        raise _nb_array_type_error()

    check_input_array(
        values, expected_ndim=2, name="Net benefit values", allow_empty=True
    )

    if values.size == 0:
        return np.array([], dtype=np.int64)

    return cast("np.ndarray", np.argmax(values, axis=1))
