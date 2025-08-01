"""Plotting functions for CEACs, CE Planes, and EVPPI surfaces."""

from typing import List, Optional, Union

import numpy as np

# Attempt to import Matplotlib, but make it optional
try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore
    Axes = None  # type: ignore

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import InputError, PlottingError
from voiage.schema import ValueArray


def _calculate_prob_ce(nb_values, n_strategies, n_wtp_points, n_samples):
    prob_ce = np.zeros((n_strategies, n_wtp_points), dtype=DEFAULT_DTYPE)

    # For each WTP threshold
    for w_idx in range(n_wtp_points):
        nb_at_wtp = nb_values[:, :, w_idx]  # (n_samples, n_strategies)
        # Identify the optimal strategy for each sample at this WTP
        optimal_strategy_indices_at_wtp = np.argmax(nb_at_wtp, axis=1)  # (n_samples,)

        # Count how many times each strategy was optimal
        for s_idx in range(n_strategies):
            prob_ce[s_idx, w_idx] = (
                np.sum(optimal_strategy_indices_at_wtp == s_idx) / n_samples
            )
    return prob_ce


def plot_ceac(
    value_array: ValueArray,
    wtp_thresholds: Union[np.ndarray, List[float]],
    strategy_names: Optional[List[str]] = None,
    xlabel: str = "Willingness-to-Pay Threshold",
    ylabel: str = "Probability Cost-Effective",
    title: str = "Cost-Effectiveness Acceptability Curve (CEAC)",
    ax: Optional[Axes] = None,
    **plot_kwargs_per_strategy: Optional[
        List[dict]
    ],  # List of dicts for each strategy's plot call
) -> Axes:
    """Plot a Cost-Effectiveness Acceptability Curve (CEAC).

    A CEAC shows the probability that each strategy is optimal (has the highest
    net benefit) across a range of willingness-to-pay (WTP) thresholds.

    Args:
        value_array (ValueArray):
            ValueArray object containing net benefit values.
            The underlying data array is expected to be 3D
            (n_samples, n_strategies, n_wtp_thresholds).
        wtp_thresholds (Union[np.ndarray, List[float]]):
            Array or list of WTP thresholds. The length must match the size of
            the last dimension of `nb_array.values` if it's 3D.
        strategy_names (Optional[List[str]]):
            Names for each strategy. If None, names from the ValueArray are used.
            Length must match `nb_array.shape[1]`.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ax (Optional[Axes]): Matplotlib Axes object to plot on.
        **plot_kwargs_per_strategy: A list of dictionaries, where each dictionary contains
                                     kwargs for the `ax.plot()` call for the corresponding strategy.
                                     If not provided, default styling is used.

    Returns
    -------
        Axes: The Matplotlib Axes object with the plot.

    Raises
    ------
        PlottingError: If Matplotlib is not installed.
        InputError: If input dimensions or lengths are mismatched.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise PlottingError(
            "Matplotlib is required for plotting functions but not installed."
        )

    nb_values = value_array.values
    if strategy_names is None:
        strategy_names = value_array.option_names

    if nb_values.ndim != 3:
        raise InputError(
            "For CEAC, nb_values must be a 3D array (samples x strategies x WTP thresholds)."
            "Ensure net benefits are calculated for each WTP.",
        )

    n_samples, n_strategies, n_wtp_points = nb_values.shape
    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)

    if len(wtp_arr) != n_wtp_points:
        raise InputError(
            f"Length of wtp_thresholds ({len(wtp_arr)}) must match the third dimension "
            f"of nb_values ({n_wtp_points}).",
        )

    if len(strategy_names) != n_strategies:
        raise InputError(
            f"Length of strategy_names ({len(strategy_names)}) must match the second dimension "
            f"of nb_values ({n_strategies}).",
        )

    if ax is None:
        fig, ax = plt.subplots()

    prob_ce = _calculate_prob_ce(nb_values, n_strategies, n_wtp_points, n_samples)

    user_plot_kwargs_list: List[dict] = plot_kwargs_per_strategy.get(
        "plot_kwargs_per_strategy", []
    )
    if len(user_plot_kwargs_list) != n_strategies and user_plot_kwargs_list:
        # Optional: Add a warning if the lengths don't match
        pass

    for s_idx in range(n_strategies):
        # Start with default kwargs
        current_kwargs: dict = {"label": strategy_names[s_idx]}

        # Get user-provided kwargs for this specific strategy
        user_kwargs: dict = {}
        if s_idx < len(user_plot_kwargs_list):
            user_kwargs = user_plot_kwargs_list[s_idx]

        # Set default color if not provided by the user
        if "color" not in user_kwargs:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            current_kwargs["color"] = prop_cycle.by_key()["color"][
                s_idx % len(prop_cycle.by_key()["color"])
            ]

        # Update with user-provided kwargs, allowing them to override defaults
        current_kwargs.update(user_kwargs)

        ax.plot(wtp_arr, prob_ce[s_idx, :], **current_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_ylim(0, 1.05)

    return ax
