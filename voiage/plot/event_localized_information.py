"""Plots derived only from event-localized information result contracts."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false, reportInvalidTypeForm=false
# pyright: reportConstantRedefinition=false, reportPossiblyUnboundVariable=false
# pyright: reportIgnoreCommentWithoutRule=false, reportUnnecessaryTypeIgnoreComment=false

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
    from voiage.methods.event_localized_information import (
        EventLocalizedInformationResult,
    )


def _axes(ax: Axes | None) -> Axes:
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required; install it with `pip install 'voiage[plotting]'`."
        )
    if ax is None:
        _figure, ax = plt.subplots()
    return ax


def plot_event_accuracy_curve(
    result: EventLocalizedInformationResult,
    title: str = "Event information value by channel accuracy",
    ax: Axes | None = None,
    **line_kwargs: Any,
) -> Axes:
    """Plot gross and net exact event VOI over evaluated channel accuracies."""
    ax = _axes(ax)
    payload = result.to_contract_dict()
    event = cast("dict[str, object]", payload["event"])
    rows = cast("list[dict[str, object]]", event["imperfect_binary_channel"])
    accuracy = [float(cast("Any", row["accuracy"])) for row in rows]
    gross = [float(cast("Any", row["gross_voi"])) for row in rows]
    net = [float(cast("Any", row["net_voi"])) for row in rows]
    kwargs: dict[str, Any] = {"marker": "o"}
    kwargs.update(line_kwargs)
    ax.plot(accuracy, gross, label="Gross VOI", **kwargs)
    ax.plot(accuracy, net, label="Net VOI", linestyle="--", **kwargs)
    ax.axvline(0.5, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Symmetric binary-channel accuracy")
    ax.set_ylabel(f"Information value ({payload['value_unit']})")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_information_density(
    result: EventLocalizedInformationResult,
    title: str = "Policy-relative information density",
    ax: Axes | None = None,
) -> Axes:
    """Plot one- or two-dimensional density atoms from an evaluated result."""
    ax = _axes(ax)
    payload = result.to_contract_dict()
    density = cast("dict[str, object]", payload["density"])
    names = cast("list[str]", density["coordinate_names"])
    units = cast("list[str]", density["coordinate_units"])
    atoms = cast("list[dict[str, object]]", density["atoms"])
    if len(names) == 1:
        x = [float(cast("list[float]", atom["coordinate"])[0]) for atom in atoms]
        y = [float(cast("Any", atom["policy_relative_density"])) for atom in atoms]
        ax.plot(x, y, marker="o", color="#4472C4")
        ax.fill_between(x, y, alpha=0.25, color="#4472C4")
        ax.set_xlabel(f"{names[0]} ({units[0]})")
    elif len(names) == 2:
        x = [float(cast("list[float]", atom["coordinate"])[0]) for atom in atoms]
        y = [float(cast("list[float]", atom["coordinate"])[1]) for atom in atoms]
        values = [float(cast("Any", atom["policy_relative_density"])) for atom in atoms]
        sizes = [
            60.0 + 240.0 * value / max(values, default=1.0)
            if max(values, default=0.0) > 0
            else 60.0
            for value in values
        ]
        points = ax.scatter(x, y, c=values, s=sizes, cmap="viridis", edgecolor="black")
        ax.figure.colorbar(points, ax=ax, label=f"Density ({payload['value_unit']})")
        ax.set_xlabel(f"{names[0]} ({units[0]})")
        ax.set_ylabel(f"{names[1]} ({units[1]})")
    else:
        raise ValueError("information-density plotting supports one or two dimensions")
    ax.set_title(title)
    if len(names) == 1:
        ax.set_ylabel(f"Policy-relative density ({payload['value_unit']})")
    return ax


__all__ = ["plot_event_accuracy_curve", "plot_information_density"]
