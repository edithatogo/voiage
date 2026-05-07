"""Plotting helpers for dominance analysis."""

from typing import Any

import numpy as np

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore

from voiage.exceptions import raise_plotting_error
from voiage.methods.dominance import DominanceResult, calculate_dominance


def plot_cost_effectiveness_plane(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
    strategy_names: list[str] | None = None,
    result: DominanceResult | None = None,
    xlabel: str = "Effect",
    ylabel: str = "Cost",
    title: str = "Cost-Effectiveness Plane",
    ax: Axes | None = None,
    **scatter_kwargs: Any,
) -> Axes:
    """Plot strategies and the cost-effectiveness frontier.

    Parameters
    ----------
    costs : numpy.ndarray or list[float]
        Strategy costs.
    effects : numpy.ndarray or list[float]
        Strategy effects.
    strategy_names : list[str], optional
        Optional strategy labels.
    result : DominanceResult, optional
        Precomputed dominance result to plot.
    xlabel : str, default="Effect"
        X-axis label.
    ylabel : str, default="Cost"
        Y-axis label.
    title : str, default="Cost-Effectiveness Plane"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    **scatter_kwargs : Any
        Keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the cost-effectiveness plane plot.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )

    dominance = result or calculate_dominance(costs, effects, strategy_names)
    if ax is None:
        _fig, ax = plt.subplots()

    colors = {
        "frontier": "black",
        "strongly_dominated": "tab:red",
        "extended_dominated": "tab:orange",
    }
    for status in sorted(set(dominance.status)):
        mask = np.asarray([item == status for item in dominance.status])
        kwargs: dict[str, Any] = {"s": 48, "label": status.replace("_", " ")}
        kwargs.update(scatter_kwargs)
        ax.scatter(
            dominance.effects[mask],
            dominance.costs[mask],
            color=colors.get(status, "tab:gray"),
            **kwargs,
        )

    if dominance.frontier_indices:
        frontier = dominance.frontier_indices
        ax.plot(
            dominance.effects[frontier],
            dominance.costs[frontier],
            color="black",
            linewidth=1.5,
            linestyle="-",
            label="frontier",
        )

    for name, effect, cost in zip(
        dominance.strategy_names,
        dominance.effects,
        dominance.costs,
        strict=True,
    ):
        ax.annotate(
            str(name), xy=(effect, cost), xytext=(4, 4), textcoords="offset points"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best")
    return ax
