# voiage/plot/ce_plane.py

"""
Cost-Effectiveness Plane plotting utilities.
"""

import matplotlib.pyplot as plt
import numpy as np

from voiage.schema import ValueArray


def plot_ce_plane(
    value_array: ValueArray,
    wtp: float,
    ax=None,
    show=True,
    **kwargs,
):
    """
    Plot the Cost-Effectiveness Plane.

    Args:
        value_array (ValueArray): A ValueArray object containing the net benefit values.
        wtp (float): The willingness-to-pay threshold.
        ax (matplotlib.axes.Axes, optional): An existing matplotlib axes to plot on.
            If None, a new figure and axes will be created. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `ax.scatter`.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    costs = (value_array.values / wtp).mean(axis=0)
    effects = value_array.values.mean(axis=0)

    ax.scatter(effects, costs, **kwargs)
    ax.set_xlabel("Incremental Effectiveness")
    ax.set_ylabel("Incremental Cost")
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(0, color="k", linestyle="--")

    if show:
        plt.show()

    return fig, ax
