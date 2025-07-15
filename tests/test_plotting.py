# tests/test_plotting.py

"""Unit tests for the plotting utilities in voiage.plot."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray


def test_plot_ce_plane():
    """Test the plot_ce_plane function."""
    nb_array = ValueArray(
        values=np.array([[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]]),
        strategy_names=["Strategy A", "Strategy B"],
    )
    analysis = DecisionAnalysis(parameters=None, values=nb_array)
    fig, ax = analysis.plot_ce_plane(wtp=20000, show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
