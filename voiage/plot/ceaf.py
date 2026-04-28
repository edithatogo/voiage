"""Plotting helpers for cost-effectiveness acceptability frontiers."""

from typing import Any

import numpy as np

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore

from voiage.exceptions import raise_plotting_error
from voiage.methods.ceaf import CEAFResult, calculate_ceaf
from voiage.schema import ValueArray


def plot_ceaf(
    value_array: ValueArray,
    wtp_thresholds: np.ndarray | list[float],
    strategy_names: list[str] | None = None,
    result: CEAFResult | None = None,
    xlabel: str = "Willingness-to-Pay Threshold",
    ylabel: str = "Probability Optimal Strategy Is Cost-Effective",
    title: str = "Cost-Effectiveness Acceptability Frontier (CEAF)",
    ax: Axes | None = None,
    show_uncertainty: bool = True,
    **plot_kwargs: object,
) -> Axes:
    """Plot a cost-effectiveness acceptability frontier.

    Parameters
    ----------
    value_array : ValueArray
        3D net-benefit surface used to build the frontier.
    wtp_thresholds : numpy.ndarray or list[float]
        Willingness-to-pay thresholds used for the x-axis.
    strategy_names : list[str], optional
        Override strategy names.
    result : CEAFResult, optional
        Precomputed CEAF result to plot.
    xlabel : str, default="Willingness-to-Pay Threshold"
        X-axis label.
    ylabel : str, default="Probability Optimal Strategy Is Cost-Effective"
        Y-axis label.
    title : str, default="Cost-Effectiveness Acceptability Frontier (CEAF)"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    show_uncertainty : bool, default=True
        Whether to draw the probability band.
    **plot_kwargs : object
        Keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the CEAF plot.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )

    ceaf_result = result or calculate_ceaf(
        value_array,
        wtp_thresholds,
        strategy_names=strategy_names,
    )
    if ax is None:
        _fig, ax = plt.subplots()

    line_kwargs: dict[str, Any] = {
        "label": "CEAF",
        "color": "black",
        "linewidth": 2.0,
    }
    line_kwargs.update(plot_kwargs)
    ax.plot(
        ceaf_result.wtp_thresholds,
        ceaf_result.acceptability_probabilities,
        **line_kwargs,
    )

    if show_uncertainty:
        ax.fill_between(
            ceaf_result.wtp_thresholds,
            ceaf_result.probability_lower,
            ceaf_result.probability_upper,
            color=str(line_kwargs.get("color", "black")),
            alpha=0.15,
            label="95% uncertainty",
        )

    switch_mask = np.r_[
        True,
        ceaf_result.optimal_strategy_indices[1:]
        != ceaf_result.optimal_strategy_indices[:-1],
    ]
    for wtp, probability, name in zip(
        ceaf_result.wtp_thresholds[switch_mask],
        ceaf_result.acceptability_probabilities[switch_mask],
        np.asarray(ceaf_result.optimal_strategy_names, dtype=object)[switch_mask],
        strict=True,
    ):
        ax.annotate(
            str(name),
            xy=(wtp, probability),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best")
    return ax
