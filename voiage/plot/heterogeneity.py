"""Plotting helpers for Value of Heterogeneity results."""

from typing import Any

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore

from voiage.exceptions import raise_plotting_error
from voiage.methods.heterogeneity import HeterogeneityResult


def plot_voh_by_subgroup(
    result: HeterogeneityResult,
    xlabel: str = "Subgroup",
    ylabel: str = "Expected Net Benefit",
    title: str = "Subgroup-Optimal Expected Net Benefit",
    ax: Axes | None = None,
    **bar_kwargs: Any,
) -> Axes:
    """Plot subgroup-specific expected net benefits."""
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )
    if ax is None:
        _fig, ax = plt.subplots()

    kwargs: dict[str, Any] = {"color": "tab:blue", "alpha": 0.8}
    kwargs.update(bar_kwargs)
    ax.bar(result.subgroup_labels, result.subgroup_expected_net_benefits, **kwargs)
    ax.axhline(
        result.overall_expected_net_benefit,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="overall optimum",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return ax
