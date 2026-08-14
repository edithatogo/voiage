"""Accessible plots for finite additive-MCDA information diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore

from voiage.exceptions import raise_plotting_error

if TYPE_CHECKING:
    from voiage.methods.mcda_information import McdaInformationResult


def _axes(ax: Axes | None) -> Axes:
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required; install it with `pip install 'voiage[plotting]'`."
        )
    if ax is None:
        _figure, ax = plt.subplots()
    return ax


def plot_mcda_information_value(
    result: McdaInformationResult,
    title: str = "Additive MCDA information value",
    ax: Axes | None = None,
    **bar_kwargs: Any,
) -> Axes:
    """Plot criterion, preference and joint gross VOI with numeric labels."""
    ax = _axes(ax)
    payload = result.to_contract_dict()
    decomposition = cast("dict[str, object]", payload["decomposition"])
    labels = ["Criterion", "Preference", "Joint"]
    values = [
        float(decomposition["criterion_gross_voi"]),
        float(decomposition["preference_gross_voi"]),
        float(decomposition["joint_gross_voi"]),
    ]
    kwargs: dict[str, Any] = {
        "color": ["#4472C4", "#70AD47", "#ED7D31"],
        "edgecolor": "black",
        "hatch": ["///", "\\\\\\", "xxx"],
    }
    kwargs.update(bar_kwargs)
    bars = ax.bar(labels, values, **kwargs)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel(f"Gross information value ({payload['aggregate_unit']})")
    ax.set_xlabel("Resolved uncertainty")
    ax.set_title(title)
    return ax


def plot_mcda_rank_acceptability(
    result: McdaInformationResult,
    title: str = "MCDA rank acceptability",
    ax: Axes | None = None,
) -> Axes:
    """Plot fractional complete-tie rank probabilities as labelled lines."""
    ax = _axes(ax)
    payload = result.to_contract_dict()
    acceptability = cast(
        "dict[str, list[float]]",
        cast("dict[str, object]", payload["rank_acceptability"])["by_alternative"],
    )
    ranks = list(range(1, len(cast("list[str]", payload["alternative_ids"])) + 1))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, alternative in enumerate(sorted(acceptability)):
        ax.plot(
            ranks,
            acceptability[alternative],
            marker=markers[index % len(markers)],
            label=alternative,
        )
    ax.set_xticks(ranks)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Probability (fractional ties)")
    ax.set_title(title)
    ax.legend(title="Alternative")
    return ax


__all__ = ["plot_mcda_information_value", "plot_mcda_rank_acceptability"]
