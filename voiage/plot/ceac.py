"""Plotting functions for CEACs, CE Planes, and EVPPI surfaces."""

import numpy as np

# Attempt to import Matplotlib, but make it optional
try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore
    Axes = None  # type: ignore

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error, raise_plotting_error
from voiage.schema import ValueArray


def _calculate_prob_ce(
    nb_values: np.ndarray,
    n_strategies: int,
    n_wtp_points: int,
    n_samples: int,
) -> np.ndarray:
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
    wtp_thresholds: np.ndarray | list[float],
    strategy_names: list[str] | None = None,
    xlabel: str = "Willingness-to-Pay Threshold",
    ylabel: str = "Probability Cost-Effective",
    title: str = "Cost-Effectiveness Acceptability Curve (CEAC)",
    ax: Axes | None = None,
    **plot_kwargs_per_strategy: object,  # List of dicts for each strategy's plot call
) -> Axes:
    """Plot a cost-effectiveness acceptability curve.

    Parameters
    ----------
    value_array : ValueArray
        3D net-benefit surface with samples, strategies, and WTP thresholds.
    wtp_thresholds : numpy.ndarray or list[float]
        Willingness-to-pay thresholds used for the x-axis.
    strategy_names : list[str], optional
        Override strategy names.
    xlabel : str, default="Willingness-to-Pay Threshold"
        X-axis label.
    ylabel : str, default="Probability Cost-Effective"
        Y-axis label.
    title : str, default="Cost-Effectiveness Acceptability Curve (CEAC)"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    **plot_kwargs_per_strategy : object
        Optional list of per-strategy keyword argument dictionaries.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the CEAC plot.

    Notes
    -----
    A CEAC shows the probability that each strategy is optimal across the
    supplied willingness-to-pay thresholds.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )

    nb_values = value_array.numpy_values
    if strategy_names is None:
        strategy_names = value_array.strategy_names

    expected_ndim = 3
    if nb_values.ndim != expected_ndim:
        raise_input_error(
            "For CEAC, nb_values must be a 3D array (samples x strategies x WTP thresholds)."
            "Ensure net benefits are calculated for each WTP.",
        )

    n_samples, n_strategies, n_wtp_points = nb_values.shape
    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)

    if len(wtp_arr) != n_wtp_points:
        raise_input_error(
            f"Length of wtp_thresholds ({len(wtp_arr)}) must match the third dimension "
            f"of nb_values ({n_wtp_points}).",
        )

    if len(strategy_names) != n_strategies:
        raise_input_error(
            f"Length of strategy_names ({len(strategy_names)}) must match the second dimension "
            f"of nb_values ({n_strategies}).",
        )

    if ax is None:
        _fig, ax = plt.subplots()

    prob_ce = _calculate_prob_ce(nb_values, n_strategies, n_wtp_points, n_samples)

    raw_plot_kwargs_list = plot_kwargs_per_strategy.get("plot_kwargs_per_strategy", [])
    if not isinstance(raw_plot_kwargs_list, list):
        raise_input_error("plot_kwargs_per_strategy must be a list of dictionaries.")

    user_plot_kwargs_list: list[dict[str, object]] = raw_plot_kwargs_list
    if len(user_plot_kwargs_list) != n_strategies and user_plot_kwargs_list:
        # Optional: Add a warning if the lengths don't match
        pass

    for s_idx in range(n_strategies):
        # Start with default kwargs
        current_kwargs: dict[str, object] = {"label": strategy_names[s_idx]}

        # Get user-provided kwargs for this specific strategy
        user_kwargs: dict[str, object] = {}
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
