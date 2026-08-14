"""Accessible tornado plot for deterministic sensitivity analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore

from voiage.exceptions import raise_plotting_error

if TYPE_CHECKING:
    from voiage.methods.deterministic_sensitivity import DeterministicSensitivityResult


def plot_deterministic_sensitivity_tornado(
    result: DeterministicSensitivityResult,
    title: str = "Deterministic Sensitivity Analysis",
    ax: Axes | None = None,
    **bar_kwargs: Any,
) -> Axes:
    """Plot ranked evaluated-grid extrema without implying interpolation.

    Bars show the minimum-to-maximum optimal metric observed on each declared
    one-way grid. Hatching and endpoint annotations keep the plot readable
    without relying on color alone.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required; install it with `pip install 'voiage[plotting]'`."
        )
    if ax is None:
        _figure, ax = plt.subplots()

    summaries = result.parameter_summaries
    positions = list(range(len(summaries)))
    kwargs: dict[str, Any] = {
        "color": "#4472C4",
        "edgecolor": "black",
        "hatch": "///",
        "alpha": 0.8,
    }
    kwargs.update(bar_kwargs)
    bars = ax.barh(
        positions,
        [item.evaluated_range for item in summaries],
        left=[item.minimum_metric for item in summaries],
        **kwargs,
    )
    for bar, item in zip(bars, summaries, strict=True):
        y = bar.get_y() + bar.get_height() / 2
        ax.annotate(
            f"{item.minimum_metric:g}",
            (item.minimum_metric, y),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
        )
        ax.annotate(
            f"{item.maximum_metric:g}",
            (item.maximum_metric, y),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )
    ax.axvline(
        result.baseline_point.optimal_metric,
        color="black",
        linestyle="--",
        label="Baseline optimum",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([item.parameter_name for item in summaries])
    ax.invert_yaxis()
    ax.set_xlabel(f"Optimal metric ({result.output_unit})")
    ax.set_ylabel("Parameter (ranked by evaluated-grid range)")
    ax.set_title(title)
    ax.legend()
    return ax
