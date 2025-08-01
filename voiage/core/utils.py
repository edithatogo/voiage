# voiage/core/utils.py

"""Utility functions for the voiage library."""

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.core.data_structures import (
    NetBenefitArray,  # Assuming NetBenefitArray is defined
)
from voiage.exceptions import DimensionMismatchError, InputError


def check_input_array(
    arr: np.ndarray,
    expected_ndim: Union[int, Sequence[int]],
    name: str = "Input array",
    expected_dtype: Optional[Union[np.dtype, type, str]] = None,
    allow_empty: bool = False,
    expected_shape: Optional[Sequence[Optional[int]]] = None,
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
        raise InputError(f"{name} must be a NumPy array. Got {type(arr)}.")

    if isinstance(expected_ndim, int):
        expected_ndim_tuple: Tuple[int, ...] = (expected_ndim,)
    else:
        expected_ndim_tuple = tuple(expected_ndim)

    if arr.ndim not in expected_ndim_tuple:
        raise DimensionMismatchError(
            f"{name} must have {expected_ndim_tuple} dimension(s). Got {arr.ndim}."
        )

    if not allow_empty and arr.size == 0:
        raise InputError(f"{name} cannot be empty.")

    dtype_to_check = expected_dtype if expected_dtype is not None else DEFAULT_DTYPE
    if isinstance(dtype_to_check, str) and dtype_to_check.lower() == "any":
        pass  # Skip dtype check
    elif arr.dtype != dtype_to_check:
        # Consider warning or allowing subtypes if more flexibility is needed.
        # For now, strict check.
        raise InputError(f"{name} must have dtype {dtype_to_check}. Got {arr.dtype}.")

    if expected_shape is not None:
        if len(expected_shape) != arr.ndim:
            raise DimensionMismatchError(
                f"Expected shape tuple length {len(expected_shape)} does not match "
                f"array ndim {arr.ndim} for {name}."
            )
        for i, (expected_dim_size, actual_dim_size) in enumerate(
            zip(expected_shape, arr.shape)
        ):
            if expected_dim_size is not None and expected_dim_size != actual_dim_size:
                raise DimensionMismatchError(
                    f"Dimension {i} of {name} has size {actual_dim_size}, "
                    f"but expected {expected_dim_size}. Shape: {arr.shape}, Expected: {expected_shape}"
                )


def calculate_net_benefit(
    costs: np.ndarray, effects: np.ndarray, wtp: Union[float, np.ndarray]
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
        expected_dtype="any",  # type: ignore
        allow_empty=False,
    )
    check_input_array(
        effects,
        expected_ndim=[1, 2],
        name="effects",
        expected_dtype="any",  # type: ignore
        allow_empty=False,
    )

    if costs.shape != effects.shape:
        raise DimensionMismatchError(
            f"Costs shape {costs.shape} and effects shape {effects.shape} must match."
        )

    if not isinstance(wtp, (float, int, np.ndarray)):
        raise InputError("WTP must be a float, int, or NumPy array.")

    wtp_arr = np.asarray(wtp, dtype=DEFAULT_DTYPE)
    if (
        wtp_arr.ndim > 2
    ):  # Allowing 0D (scalar), 1D (thresholds), 2D (e.g. samples x thresholds)
        raise DimensionMismatchError(
            "WTP array cannot have more than 2 dimensions for this function."
        )
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

    return nmb.astype(DEFAULT_DTYPE, copy=False)  # type: ignore


def get_optimal_strategy_index(
    nb_array: Union[np.ndarray, NetBenefitArray],
) -> np.ndarray[Any, np.dtype[np.int64]]:
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
    if isinstance(nb_array, NetBenefitArray):
        values = nb_array.values
    elif isinstance(nb_array, np.ndarray):
        values = nb_array
    else:
        raise InputError("nb_array must be a NumPy array or NetBenefitArray instance.")

    check_input_array(
        values, expected_ndim=2, name="Net benefit values", allow_empty=True
    )

    if values.size == 0:
        return np.array([], dtype=np.int64)

    return np.argmax(values, axis=1).astype(np.int64)
