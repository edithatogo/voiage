# voiage/core/utils.py

"""
Utility functions for the voiage library.

This module contains helper functions that are used across various parts of
the library, such as input validation, common mathematical operations not
specific to a single VOI method, or data transformations.
"""

from typing import Optional, Sequence, Union

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.schema import (
    ValueArray,
)
from voiage.exceptions import DimensionMismatchError, InputError


def check_input_array(
    arr: np.ndarray,
    expected_ndim: Union[int, Sequence[int]],
    name: str = "Input array",
    expected_dtype: Optional[
        np.dtype
    ] = None,  # Allow checking specific dtype if needed
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
        expected_ndim_tuple = (expected_ndim,)
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

    return nmb.astype(DEFAULT_DTYPE, copy=False)


def get_optimal_strategy_index(
    nb_array: Union[np.ndarray, ValueArray], axis: int = 1
) -> np.ndarray:
    """Determine the index of the optimal strategy for each sample.

    The optimal strategy is the one with the highest net benefit.

    Args:
        nb_array (Union[np.ndarray, ValueArray]): A 2D NumPy array of net benefits
            (samples x strategies) or a ValueArray instance.
        axis (int): The axis along which to find the maximum. Defaults to 1 (strategies).

    Returns
    -------
        np.ndarray: A 1D array containing the index of the optimal strategy for each sample.
    """
    if isinstance(nb_array, ValueArray):
        values = nb_array.values
    elif isinstance(nb_array, np.ndarray):
        values = nb_array
    else:
        raise InputError("nb_array must be a NumPy array or ValueArray instance.")

    check_input_array(
        values, expected_ndim=2, name="Net benefit values", allow_empty=False
    )

    if axis != 1:
        # This function is specifically for finding optimal strategy across strategy axis.
        # Generalizing axis might be confusing for this particular utility.
        raise ValueError(
            "`axis` must be 1 for finding optimal strategy index across strategies."
        )

    if values.shape[axis] == 0:  # No strategies to choose from
        # Return an array of NaNs or raise error, depends on desired behavior
        return np.full(
            values.shape[0], np.nan, dtype=np.intp
        )  # np.intp usually for indices

    optimal_indices = np.argmax(values, axis=axis)
    return optimal_indices


def compute_incremental_net_benefit(
    nb_array: Union[np.ndarray, ValueArray], comparator_index: int = 0
) -> np.ndarray:
    """Compute incremental net benefit (INB) relative to a comparator strategy.

    Args:
        nb_array (Union[np.ndarray, ValueArray]): A 2D NumPy array of net benefits
            (samples x strategies) or a ValueArray instance.
        comparator_index (int): Index of the comparator strategy. Defaults to 0.

    Returns
    -------
        np.ndarray: A 2D array of incremental net benefits. Shape (n_samples, n_strategies - 1).
                    The column corresponding to the comparator is removed.

    Raises
    ------
        InputError: If comparator_index is out of bounds.
    """
    if isinstance(nb_array, ValueArray):
        values = nb_array.values
    elif isinstance(nb_array, np.ndarray):
        values = nb_array
    else:
        raise InputError("nb_array must be a NumPy array or ValueArray instance.")

    check_input_array(
        values, expected_ndim=2, name="Net benefit values", allow_empty=False
    )

    n_strategies = values.shape[1]
    if not (0 <= comparator_index < n_strategies):
        raise InputError(
            f"Comparator index {comparator_index} is out of bounds for {n_strategies} strategies."
        )

    comparator_nb = values[
        :, comparator_index, np.newaxis
    ]  # Keep it as a column vector
    inb = values - comparator_nb

    # Remove the comparator column itself (which will be all zeros)
    inb_without_comparator = np.delete(inb, comparator_index, axis=1)

    return inb_without_comparator


# --- Higher-order utility for parameter sampling if needed ---
# def generate_parameter_samples(
#     distributions: Dict[str, Callable[..., np.ndarray]], # e.g., {"param_a": np.random.normal}
#     dist_args: Dict[str, dict], # e.g., {"param_a": {"loc": 0, "scale": 1}}
#     n_samples: int
# ) -> 'PSASample': # Forward reference for PSASample
#     """
#     Generates parameter samples from specified distributions.
#     This is a simplified example. More robust sampling might come from PyMC, etc.
#     """
#     from pyvoi.core.data_structures import PSASample # Local import to avoid circularity at load time
#     params_dict = {}
#     for param_name, dist_func in distributions.items():
#         args = dist_args.get(param_name, {})
#         try:
#             # Ensure n_samples (or size) is passed correctly
#             if 'size' not in args and 'shape' not in args: # common param names for sample count
#                 args_with_size = {**args, 'size': n_samples} # some dists use 'size'
#             else: # if size/shape is already there, assume it's correct or will be handled by dist_func
#                 args_with_size = args
#
#             samples = dist_func(**args_with_size)
#
#             # Ensure samples are 1D of length n_samples
#             if samples.ndim == 0 and n_samples == 1: # scalar result for 1 sample
#                 samples = np.array([samples], dtype=DEFAULT_DTYPE)
#             elif samples.shape != (n_samples,):
#                 # Try to reshape if total size matches, otherwise error
#                 if samples.size == n_samples:
#                     samples = samples.reshape(n_samples)
#                 else:
#                     raise DimensionMismatchError(
#                         f"Generated samples for '{param_name}' have shape {samples.shape}, "
#                         f"expected ({n_samples},). Total size {samples.size}."
#                     )
#             params_dict[param_name] = samples.astype(DEFAULT_DTYPE, copy=False)
#         except Exception as e:
#             raise CalculationError(f"Error sampling for parameter '{param_name}': {e}") from e
#
#     return PSASample(parameters=params_dict)


if __name__ == "__main__":
    print("--- Testing Utility functions ---")

    # Test check_input_array
    try:
        check_input_array(
            np.array([[1, 2], [3, 4]]), expected_ndim=2, name="Test Array"
        )
        print("check_input_array basic test PASSED.")
        check_input_array(np.array([1, 2, 3]), expected_ndim=1, expected_shape=(3,))
        print("check_input_array with shape PASSED.")
        check_input_array(np.array([1, 2, 3]), expected_ndim=1, expected_shape=(None,))
        print("check_input_array with None in shape PASSED.")
        check_input_array(
            np.array([1.0, 2.0]), expected_ndim=1, expected_dtype=np.float64
        )
        print("check_input_array with dtype PASSED.")
        check_input_array(np.array([]), expected_ndim=1, allow_empty=True)
        print("check_input_array with allow_empty PASSED.")

        fail_count = 0
        try:
            check_input_array(np.array([1, 2]), expected_ndim=2)
        except DimensionMismatchError:
            print("Caught expected ndim mismatch.")
        else:
            fail_count += 1

        try:
            check_input_array(np.array([[1, 2]]), expected_shape=(1, 3))
        except DimensionMismatchError:
            print("Caught expected shape mismatch.")
        else:
            fail_count += 1

        try:
            check_input_array(
                np.array([1]),
                expected_ndim=1,
                allow_empty=False,
                expected_dtype=np.int32,
            )
        except InputError:
            print("Caught expected dtype mismatch.")  # Current default is float64
        else:
            fail_count += 1

        if fail_count > 0:
            print(f"check_input_array negative tests FAILED {fail_count} times.")
        else:
            print("check_input_array negative tests PASSED.")

    except Exception as e:
        print(f"check_input_array test FAILED: {e}")

    # Test calculate_net_benefit
    costs_arr = np.array([[10, 20], [15, 25]], dtype=DEFAULT_DTYPE)
    effects_arr = np.array([[1, 1.5], [1.2, 1.8]], dtype=DEFAULT_DTYPE)
    wtp_scalar = 100.0

    nmb_scalar_wtp = calculate_net_benefit(costs_arr, effects_arr, wtp_scalar)
    expected_nmb_scalar = (effects_arr * wtp_scalar) - costs_arr
    np.testing.assert_allclose(nmb_scalar_wtp, expected_nmb_scalar)
    print("calculate_net_benefit with scalar WTP PASSED.")

    wtp_array = np.array([50, 100, 150], dtype=DEFAULT_DTYPE)  # (K,)
    nmb_array_wtp = calculate_net_benefit(costs_arr, effects_arr, wtp_array)
    # Expected shape (N, S, K) -> (2, 2, 3)
    expected_nmb_array_manual = np.zeros((2, 2, 3), dtype=DEFAULT_DTYPE)
    for k_idx, w_val in enumerate(wtp_array):
        expected_nmb_array_manual[:, :, k_idx] = (effects_arr * w_val) - costs_arr
    np.testing.assert_allclose(nmb_array_wtp.shape, (2, 2, 3))
    np.testing.assert_allclose(nmb_array_wtp, expected_nmb_array_manual)
    print("calculate_net_benefit with array WTP (N,S) x (K,) -> (N,S,K) PASSED.")

    costs_1d = np.array([10, 15], dtype=DEFAULT_DTYPE)  # (N,)
    effects_1d = np.array([1, 1.2], dtype=DEFAULT_DTYPE)  # (N,)
    nmb_1d_array_wtp = calculate_net_benefit(costs_1d, effects_1d, wtp_array)
    # Expected shape (N,K) -> (2,3)
    expected_nmb_1d_array_manual = np.zeros((2, 3), dtype=DEFAULT_DTYPE)
    for k_idx, w_val in enumerate(wtp_array):
        expected_nmb_1d_array_manual[:, k_idx] = (effects_1d * w_val) - costs_1d
    np.testing.assert_allclose(nmb_1d_array_wtp.shape, (2, 3))
    np.testing.assert_allclose(nmb_1d_array_wtp, expected_nmb_1d_array_manual)
    print("calculate_net_benefit with array WTP (N,) x (K,) -> (N,K) PASSED.")

    # Test get_optimal_strategy_index
    nb_vals = np.array([[10, 30, 20], [50, 40, 45]], dtype=DEFAULT_DTYPE)
    optimal_idx = get_optimal_strategy_index(nb_vals)
    np.testing.assert_allclose(optimal_idx, np.array([1, 0]))
    print("get_optimal_strategy_index PASSED.")

    # Test compute_incremental_net_benefit
    inb = compute_incremental_net_benefit(nb_vals, comparator_index=0)
    expected_inb = np.array(
        [[20, 10], [-10, -5]], dtype=DEFAULT_DTYPE
    )  # (30-10, 20-10), (40-50, 45-50)
    np.testing.assert_allclose(inb, expected_inb)
    print("compute_incremental_net_benefit PASSED.")

    inb_comp1 = compute_incremental_net_benefit(nb_vals, comparator_index=1)
    expected_inb_comp1 = np.array(
        [[-20, -10], [10, 5]], dtype=DEFAULT_DTYPE
    )  # (10-30, 20-30), (50-40, 45-40)
    np.testing.assert_allclose(inb_comp1, expected_inb_comp1)
    print("compute_incremental_net_benefit with comparator 1 PASSED.")

    print("\n--- Utils Testing Done ---")
