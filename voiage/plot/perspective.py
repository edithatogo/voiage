"""Plotting helpers for Value of Perspective analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore

from voiage.exceptions import raise_plotting_error

if TYPE_CHECKING:
    from voiage.methods.perspective import ValueOfPerspectiveResult


def plot_perspective_regret(
    result: ValueOfPerspectiveResult,
    title: str = "Cross-Perspective Regret",
    ax: Axes | None = None,
    **imshow_kwargs: Any,
) -> Axes:
    """Plot the Value of Perspective regret matrix.

    Parameters
    ----------
    result : ValueOfPerspectiveResult
        Result from :func:`voiage.methods.perspective.value_of_perspective`.
    title : str, default="Cross-Perspective Regret"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    **imshow_kwargs : object
        Additional keyword arguments passed to ``Axes.imshow``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the regret heatmap.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required; install it with `pip install 'voiage[plotting]'`."
        )
    if ax is None:
        _fig, ax = plt.subplots()

    kwargs: dict[str, Any] = {"cmap": "viridis"}
    kwargs.update(imshow_kwargs)
    image = ax.imshow(result.regret_matrix, **kwargs)
    ax.set_xticks(np.arange(len(result.perspective_labels)))
    ax.set_yticks(np.arange(len(result.perspective_labels)))
    ax.set_xticklabels(result.perspective_labels, rotation=45, ha="right")
    ax.set_yticklabels(result.perspective_labels)
    ax.set_xlabel("Decision rule perspective")
    ax.set_ylabel("Evaluation perspective")
    ax.set_title(title)

    for row_idx in range(result.regret_matrix.shape[0]):
        for column_idx in range(result.regret_matrix.shape[1]):
            ax.text(
                column_idx,
                row_idx,
                f"{result.regret_matrix[row_idx, column_idx]:.2g}",
                ha="center",
                va="center",
                color="white",
            )

    ax.figure.colorbar(image, ax=ax, label="Regret")
    return ax
