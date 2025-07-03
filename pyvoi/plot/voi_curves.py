# pyvoi/plot/voi_curves.py

"""Plotting functions for various VOI curves."""

from typing import List, Optional, Union

import numpy as np

# Attempt to import Matplotlib, but make it optional
try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore
    Axes = None  # type: ignore

from pyvoi.config import DEFAULT_DTYPE
from pyvoi.exceptions import InputError, PlottingError


def plot_evpi_vs_wtp(
    evpi_values: Union[np.ndarray, List[float]],
    wtp_thresholds: Union[np.ndarray, List[float]],
    xlabel: str = "Willingness-to-Pay Threshold",
    ylabel: str = "EVPI",
    title: str = "Expected Value of Perfect Information vs. WTP",
    ax: Optional[Axes] = None,
    **plot_kwargs,
) -> Axes:
    """Plot EVPI against a range of Willingness-to-Pay (WTP) thresholds.

    Args:
        evpi_values (Union[np.ndarray, List[float]]):
            Array or list of EVPI values corresponding to each WTP threshold.
        wtp_thresholds (Union[np.ndarray, List[float]]):
            Array or list of WTP thresholds. Must be the same length as evpi_values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ax (Optional[Axes]): Matplotlib Axes object to plot on. If None, a new
                             figure and axes are created.
        **plot_kwargs: Additional keyword arguments passed to `ax.plot()`.

    Returns
    -------
        Axes: The Matplotlib Axes object with the plot.

    Raises
    ------
        PlottingError: If Matplotlib is not installed.
        InputError: If input array lengths do not match.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise PlottingError(
            "Matplotlib is required for plotting functions but not installed."
        )

    evpi_arr = np.asarray(evpi_values, dtype=DEFAULT_DTYPE)
    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)

    if evpi_arr.ndim != 1 or wtp_arr.ndim != 1:
        raise InputError("evpi_values and wtp_thresholds must be 1-dimensional.")
    if len(evpi_arr) != len(wtp_arr):
        raise InputError("Length of evpi_values must match length of wtp_thresholds.")
    if len(evpi_arr) == 0:
        # print("Warning: No data to plot for EVPI vs WTP.")
        # Create an empty plot or raise error? For now, allow empty plot.
        pass

    if ax is None:
        fig, ax = plt.subplots()  # type: ignore

    ax.plot(wtp_arr, evpi_arr, **plot_kwargs)  # type: ignore
    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.grid(True, linestyle=":", alpha=0.7)  # type: ignore

    return ax  # type: ignore


def plot_evsi_vs_sample_size(
    evsi_values: Union[np.ndarray, List[float]],
    sample_sizes: Union[np.ndarray, List[int], List[float]],
    enbs_values: Optional[Union[np.ndarray, List[float]]] = None,
    research_costs: Optional[Union[np.ndarray, List[float]]] = None,
    xlabel: str = "Sample Size (per arm or total)",
    ylabel_evsi: str = "EVSI",
    ylabel_enbs: str = "ENBS / Cost",
    title: str = "Expected Value of Sample Information vs. Sample Size",
    ax: Optional[Axes] = None,
    plot_evsi_kwargs: Optional[dict] = None,
    plot_enbs_kwargs: Optional[dict] = None,
    plot_cost_kwargs: Optional[dict] = None,
) -> Axes:
    """Plot EVSI and optionally ENBS and research costs against sample sizes.

    Args:
        evsi_values (Union[np.ndarray, List[float]]): EVSI values.
        sample_sizes (Union[np.ndarray, List[int], List[float]]): Corresponding sample sizes.
        enbs_values (Optional[Union[np.ndarray, List[float]]]): Optional ENBS values.
        research_costs (Optional[Union[np.ndarray, List[float]]]): Optional research costs.
        xlabel (str): X-axis label.
        ylabel_evsi (str): Y-axis label for EVSI (if ENBS/costs not plotted on same axis).
        ylabel_enbs (str): Y-axis label if ENBS/costs are plotted (can be secondary y-axis).
        title (str): Plot title.
        ax (Optional[Axes]): Matplotlib Axes to plot on. If None, new figure/axes created.
        plot_evsi_kwargs (Optional[dict]): Kwargs for EVSI plot line.
        plot_enbs_kwargs (Optional[dict]): Kwargs for ENBS plot line.
        plot_cost_kwargs (Optional[dict]): Kwargs for research cost plot line.

    Returns
    -------
        Axes: The Matplotlib Axes object.

    Raises
    ------
        PlottingError: If Matplotlib is not installed.
        InputError: If input array lengths do not match.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise PlottingError(
            "Matplotlib is required for plotting functions but not installed."
        )

    evsi_arr = np.asarray(evsi_values, dtype=DEFAULT_DTYPE)
    ss_arr = np.asarray(sample_sizes)  # Can be int or float

    if evsi_arr.ndim != 1 or ss_arr.ndim != 1:
        raise InputError("evsi_values and sample_sizes must be 1-dimensional.")
    if len(evsi_arr) != len(ss_arr):
        raise InputError("Length of evsi_values must match length of sample_sizes.")

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
        fig, ax1 = plt.subplots()  # type: ignore
    else:
        ax1 = ax

    ln1 = ax1.plot(ss_arr, evsi_arr, **_plot_evsi_kwargs)  # type: ignore
    ax1.set_xlabel(xlabel)  # type: ignore
    ax1.set_ylabel(ylabel_evsi)  # type: ignore
    ax1.grid(True, linestyle=":", alpha=0.7)  # type: ignore

    lines = ln1
    labels = [plot_line.get_label() for plot_line in lines]

    # Plot ENBS and Costs if provided, potentially on a secondary y-axis if scales differ significantly
    if enbs_values is not None or research_costs is not None:
        # Determine if secondary axis is needed based on scale relative to EVSI
        use_secondary_axis = False
        if enbs_values is not None:
            enbs_arr = np.asarray(enbs_values, dtype=DEFAULT_DTYPE)
            if len(enbs_arr) != len(ss_arr):
                raise InputError("Length of enbs_values mismatch.")
            # Simple heuristic for secondary axis: if max ENBS is very different from max EVSI
            if (
                len(evsi_arr) > 0
                and len(enbs_arr) > 0
                and (
                    np.max(np.abs(evsi_arr)) > 0
                    and (
                        np.max(np.abs(enbs_arr)) / np.max(np.abs(evsi_arr)) < 0.1
                        or np.max(np.abs(enbs_arr)) / np.max(np.abs(evsi_arr)) > 10
                    )
                )
            ):
                # This heuristic might need refinement
                # use_secondary_axis = True # Decided against for simplicity unless explicitly requested
                pass

        ax2 = ax1  # Default to same axis
        if use_secondary_axis:
            ax2 = ax1.twinx()  # type: ignore
            ax2.set_ylabel(ylabel_enbs)  # type: ignore
        elif (
            enbs_values is not None or research_costs is not None
        ):  # If plotting more than EVSI on ax1
            ax1.set_ylabel(f"{ylabel_evsi} / {ylabel_enbs}")

        if enbs_values is not None:
            enbs_arr = np.asarray(
                enbs_values, dtype=DEFAULT_DTYPE
            )  # Re-assert for safety
            ln2 = ax2.plot(ss_arr, enbs_arr, **_plot_enbs_kwargs)  # type: ignore
            lines.extend(ln2)
            labels.extend([plot_line.get_label() for plot_line in ln2])

        if research_costs is not None:
            cost_arr = np.asarray(research_costs, dtype=DEFAULT_DTYPE)
            if len(cost_arr) != len(ss_arr):
                raise InputError("Length of research_costs mismatch.")
            ln3 = ax2.plot(ss_arr, cost_arr, **_plot_cost_kwargs)  # type: ignore
            lines.extend(ln3)
            labels.extend([plot_line.get_label() for plot_line in ln3])

    ax1.legend(lines, labels, loc="best")  # type: ignore
    ax1.set_title(title)  # type: ignore

    # Ensure layout is tight if a secondary axis was created
    # if use_secondary_axis and fig is not None: # fig would be defined if ax was None
    # fig.tight_layout() # Often helpful with twin axes

    return ax1  # type: ignore


# Future plots:
# - plot_dynamic_voi (VOI metrics over time steps)
# - plot_evppi_vs_parameters (e.g., Tornado diagram if applicable, or heatmap for 2 params)

if __name__ == "__main__":
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
