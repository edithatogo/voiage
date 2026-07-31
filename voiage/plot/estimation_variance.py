"""Accessible plotting for estimation-focused variance reduction."""

# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnusedCallResult=false

from __future__ import annotations

from typing import TYPE_CHECKING

from voiage.exceptions import raise_plotting_error

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from voiage.contracts.estimation import EstimationVarianceResult


def plot_estimation_variance(
    result: EstimationVarianceResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot current and expected posterior uncertainty with direct labels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        raise_plotting_error(
            "Matplotlib is required; install it with `pip install 'voiage[plotting]'`."
        )

    if ax is None:
        _figure, ax = plt.subplots()
    labels = ["Current information", "After information"]
    values = [result.prior_functional, result.expected_posterior_functional]
    bars = ax.bar(
        labels,
        values,
        color=["#0072B2", "#E69F00"],
        edgecolor="black",
    )
    bars[1].set_hatch("//")
    _ = ax.bar_label(bars, fmt="%.4g", padding=3)
    ax.set_ylabel(f"Variance functional ({result.functional_units})")
    ax.set_title(f"Estimation uncertainty: {result.target.target_id}")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.text(
        0.5,
        0.98,
        f"Reduction: {result.absolute_reduction:.4g}",
        ha="center",
        va="top",
        transform=ax.transAxes,
    )
    return ax
