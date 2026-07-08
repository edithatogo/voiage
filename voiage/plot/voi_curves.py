# voiage/plot/voi_curves.py

"""Plotting functions for various VOI curves."""

import numpy as np

# Attempt to import Matplotlib, but make it optional
try:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    Axes = None  # type: ignore
    Axes3D = None  # type: ignore

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error, raise_plotting_error


def plot_evpi_vs_wtp(
    evpi_values: np.ndarray | list[float],
    wtp_thresholds: np.ndarray | list[float],
    xlabel: str = "Willingness-to-Pay Threshold",
    ylabel: str = "EVPI",
    title: str = "Expected Value of Perfect Information vs. WTP",
    ax: Axes | None = None,
    **plot_kwargs: object,
) -> Axes:
    """Plot EVPI against willingness-to-pay thresholds.

    Parameters
    ----------
    evpi_values : numpy.ndarray or list[float]
        EVPI values for each threshold.
    wtp_thresholds : numpy.ndarray or list[float]
        Willingness-to-pay thresholds.
    xlabel : str, default="Willingness-to-Pay Threshold"
        X-axis label.
    ylabel : str, default="EVPI"
        Y-axis label.
    title : str, default="Expected Value of Perfect Information vs. WTP"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    **plot_kwargs : object
        Keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the EVPI curve.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )

    evpi_arr = np.asarray(evpi_values, dtype=DEFAULT_DTYPE)
    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)

    if evpi_arr.ndim != 1 or wtp_arr.ndim != 1:
        raise_input_error("evpi_values and wtp_thresholds must be 1-dimensional.")
    if len(evpi_arr) != len(wtp_arr):
        raise_input_error("Length of evpi_values must match length of wtp_thresholds.")
    if len(evpi_arr) == 0:
        # print("Warning: No data to plot for EVPI vs WTP.")
        # Create an empty plot or raise error? For now, allow empty plot.
        pass

    if ax is None:
        _fig, ax = plt.subplots()  # type: ignore

    ax.plot(wtp_arr, evpi_arr, **plot_kwargs)  # type: ignore
    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.grid(True, linestyle=":", alpha=0.7)  # type: ignore

    return ax  # type: ignore


def _plot_enbs_and_costs(
    ax: Axes,
    ss_arr: np.ndarray,
    enbs_values: np.ndarray | list[float] | None,
    research_costs: np.ndarray | list[float] | None,
    ylabel_enbs: str,
    plot_enbs_kwargs: dict[str, object],
    plot_cost_kwargs: dict[str, object],
) -> tuple[list[object], list[str]]:
    lines: list[object] = []
    labels: list[str] = []

    # Plot ENBS and Costs if provided, potentially on a secondary y-axis if scales differ significantly
    if enbs_values is not None or research_costs is not None:
        if enbs_values is not None:
            enbs_arr = np.asarray(enbs_values, dtype=DEFAULT_DTYPE)
            if len(enbs_arr) != len(ss_arr):
                raise_input_error("Length of enbs_values mismatch.")

        ax2 = ax
        if enbs_values is not None or research_costs is not None:
            ax.set_ylabel(f"{ax.get_ylabel()} / {ylabel_enbs}")

        if enbs_values is not None:
            enbs_arr = np.asarray(
                enbs_values, dtype=DEFAULT_DTYPE
            )  # Re-assert for safety
            ln2 = ax2.plot(ss_arr, enbs_arr, **plot_enbs_kwargs)  # type: ignore
            lines.extend(ln2)
            labels.extend([plot_line.get_label() for plot_line in ln2])

        if research_costs is not None:
            cost_arr = np.asarray(research_costs, dtype=DEFAULT_DTYPE)
            if len(cost_arr) != len(ss_arr):
                raise_input_error("Length of research_costs mismatch.")
            ln3 = ax2.plot(ss_arr, cost_arr, **plot_cost_kwargs)  # type: ignore
            lines.extend(ln3)
            labels.extend([plot_line.get_label() for plot_line in ln3])

    return lines, labels


def plot_evsi_vs_sample_size(
    evsi_values: np.ndarray | list[float],
    sample_sizes: np.ndarray | list[int] | list[float],
    enbs_values: np.ndarray | list[float] | None = None,
    research_costs: np.ndarray | list[float] | None = None,
    xlabel: str = "Sample Size (per arm or total)",
    ylabel_evsi: str = "EVSI",
    ylabel_enbs: str = "ENBS / Cost",
    title: str = "Expected Value of Sample Information vs. Sample Size",
    ax: Axes | None = None,
    plot_evsi_kwargs: dict | None = None,
    plot_enbs_kwargs: dict | None = None,
    plot_cost_kwargs: dict | None = None,
) -> Axes:
    """Plot EVSI against sample size with optional ENBS and cost curves.

    Parameters
    ----------
    evsi_values : numpy.ndarray or list[float]
        EVSI values.
    sample_sizes : numpy.ndarray or list[int] or list[float]
        Sample sizes corresponding to the EVSI values.
    enbs_values : numpy.ndarray or list[float], optional
        Optional ENBS values.
    research_costs : numpy.ndarray or list[float], optional
        Optional research costs.
    xlabel : str, default="Sample Size (per arm or total)"
        X-axis label.
    ylabel_evsi : str, default="EVSI"
        Y-axis label for EVSI.
    ylabel_enbs : str, default="ENBS / Cost"
        Y-axis label for ENBS and cost curves.
    title : str, default="Expected Value of Sample Information vs. Sample Size"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    plot_evsi_kwargs : dict, optional
        Keyword arguments for the EVSI line.
    plot_enbs_kwargs : dict, optional
        Keyword arguments for the ENBS line.
    plot_cost_kwargs : dict, optional
        Keyword arguments for the research cost line.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the EVSI plot.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )

    evsi_arr = np.asarray(evsi_values, dtype=DEFAULT_DTYPE)
    ss_arr = np.asarray(sample_sizes)  # Can be int or float

    if evsi_arr.ndim != 1 or ss_arr.ndim != 1:
        raise_input_error("evsi_values and sample_sizes must be 1-dimensional.")
    if len(evsi_arr) != len(ss_arr):
        raise_input_error("Length of evsi_values must match length of sample_sizes.")

    _plot_evsi_kwargs = {"label": "EVSI", "color": "blue"}
    if plot_evsi_kwargs:
        _plot_evsi_kwargs.update(plot_evsi_kwargs)

    _plot_enbs_kwargs = {"label": "ENBS", "color": "green", "linestyle": "--"}
    if plot_enbs_kwargs:
        _plot_enbs_kwargs.update(plot_enbs_kwargs)

    _plot_cost_kwargs = {"label": "Research Cost", "color": "red", "linestyle": ":"}
    if plot_cost_kwargs:
        _plot_cost_kwargs.update(plot_cost_kwargs)

    if ax is None:
        _fig, ax1 = plt.subplots()  # type: ignore
    else:
        ax1 = ax

    ln1 = ax1.plot(ss_arr, evsi_arr, **_plot_evsi_kwargs)  # type: ignore
    ax1.set_xlabel(xlabel)  # type: ignore
    ax1.set_ylabel(ylabel_evsi)  # type: ignore
    ax1.grid(True, linestyle=":", alpha=0.7)  # type: ignore

    lines = ln1
    labels = [plot_line.get_label() for plot_line in lines]

    enbs_lines, enbs_labels = _plot_enbs_and_costs(
        ax1,
        ss_arr,
        enbs_values,
        research_costs,
        ylabel_enbs,
        _plot_enbs_kwargs,
        _plot_cost_kwargs,
    )
    lines.extend(enbs_lines)
    labels.extend(enbs_labels)

    ax1.legend(lines, labels, loc="best")  # type: ignore
    ax1.set_title(title)  # type: ignore

    # Ensure layout is tight if a secondary axis was created
    # if use_secondary_axis and fig is not None: # fig would be defined if ax was None
    # fig.tight_layout() # Often helpful with twin axes

    return ax1  # type: ignore


def plot_evppi_surface(
    evppi_values: np.ndarray,
    param_names: list[str],
    wtp_thresholds: np.ndarray | list[float],
    xlabel: str = "Parameter",
    ylabel: str = "Willingness-to-Pay Threshold",
    zlabel: str = "EVPPI",
    title: str = "EVPPI Surface",
    ax: Axes | None = None,
    **plot_kwargs: object,
) -> Axes:
    """Plot a 3D EVPPI surface.

    Parameters
    ----------
    evppi_values : numpy.ndarray
        2D array of EVPPI values with shape ``(n_params, n_wtp_thresholds)``.
    param_names : list[str]
        Parameter names.
    wtp_thresholds : numpy.ndarray or list[float]
        Willingness-to-pay thresholds.
    xlabel : str, default="Parameter"
        X-axis label.
    ylabel : str, default="Willingness-to-Pay Threshold"
        Y-axis label.
    zlabel : str, default="EVPPI"
        Z-axis label.
    title : str, default="EVPPI Surface"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    **plot_kwargs : object
        Keyword arguments forwarded to ``ax.plot_surface``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the EVPPI surface.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise_plotting_error(
            "Matplotlib is required for plotting functions but not installed."
        )

    evppi_arr = np.asarray(evppi_values, dtype=DEFAULT_DTYPE)
    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)

    expected_ndim = 2
    if evppi_arr.ndim != expected_ndim:
        raise_input_error(
            "evppi_values must be a 2D array (n_params x n_wtp_thresholds)."
        )
    if len(param_names) != evppi_arr.shape[0]:
        raise_input_error(
            "Length of param_names must match the first dimension of evppi_values."
        )
    if len(wtp_arr) != evppi_arr.shape[1]:
        raise_input_error(
            "Length of wtp_thresholds must match the second dimension of evppi_values."
        )

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    x, y = np.meshgrid(np.arange(len(param_names)), wtp_arr)
    ax.plot_surface(x, y, evppi_arr.T, **plot_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45)

    return ax


# Future plots:
# - plot_dynamic_voi (VOI metrics over time steps)
# - plot_evppi_vs_parameters (e.g., Tornado diagram if applicable, or heatmap for 2 params)

if __name__ == "__main__":  # pragma: no cover
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed. Skipping plot generation examples.")
    else:
        print("--- Testing voi_curves.py ---")

        # Test plot_evpi_vs_wtp
        wtps = np.linspace(0, 100000, 20, dtype=DEFAULT_DTYPE)
        evpis = 5000 * (1 - np.exp(-wtps / 30000))  # Example EVPI curve

        fig1, ax1 = plt.subplots()  # type: ignore
        plot_evpi_vs_wtp(evpis, wtps, ax=ax1, label="EVPI Data", color="purple")
        ax1.legend()  # type: ignore
        # plt.show() # Uncomment to display plot during testing
        print("plot_evpi_vs_wtp example generated.")
        plt.close(fig1)  # type: ignore

        # Test plot_evsi_vs_sample_size
        sample_sizes_test = np.array([10, 50, 100, 200, 500, 1000], dtype=DEFAULT_DTYPE)
        evsi_test_vals = 2000 * (
            1 - np.exp(-sample_sizes_test / 200)
        )  # EVSI increases with N
        costs_test_vals = 50 + 1.5 * sample_sizes_test  # Cost increases linearly with N
        enbs_test_vals = evsi_test_vals - costs_test_vals

        fig2, ax2_main = plt.subplots()  # type: ignore
        plot_evsi_vs_sample_size(
            evsi_values=evsi_test_vals,
            sample_sizes=sample_sizes_test,
            enbs_values=enbs_test_vals,
            research_costs=costs_test_vals,
            ax=ax2_main,
        )
        # plt.show() # Uncomment to display plot
        print("plot_evsi_vs_sample_size example generated.")
        plt.close(fig2)  # type: ignore

        # Test with only EVSI
        fig3, ax3 = plt.subplots()  # type: ignore
        plot_evsi_vs_sample_size(
            evsi_values=evsi_test_vals,
            sample_sizes=sample_sizes_test,
            ax=ax3,
            title="EVSI Only vs. Sample Size",
        )
        # plt.show()
        print("plot_evsi_vs_sample_size (EVSI only) example generated.")
        plt.close(fig3)  # type: ignore

        print(
            "--- voi_curves.py tests completed (plots generated if Matplotlib available) ---"
        )
