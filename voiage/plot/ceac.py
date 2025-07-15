# voiage/plot/ceac.py

"""Plotting functions for CEACs, CE Planes, and EVPPI surfaces."""

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

from voiage.config import DEFAULT_DTYPE
from voiage.core.data_structures import NetBenefitArray
from voiage.exceptions import InputError, PlottingError


def _calculate_prob_ce(nb_values, n_strategies, n_wtp_points, n_samples):
    prob_ce = np.zeros((n_strategies, n_wtp_points), dtype=DEFAULT_DTYPE)

    # For each WTP threshold
    for w_idx in range(n_wtp_points):
        nb_at_wtp = nb_values[:, :, w_idx]  # (n_samples, n_strategies)
        # Identify the optimal strategy for each sample at this WTP
        optimal_strategy_indices_at_wtp = np.argmax(nb_at_wtp, axis=1)  # (n_samples,)

        # Count how many times each strategy was optimal
        for s_idx in range(n_strategies):
            prob_ce[s_idx, w_idx] = (
                np.sum(optimal_strategy_indices_at_wtp == s_idx) / n_samples
            )
    return prob_ce


def plot_ceac(
    nb_array: Union[np.ndarray, NetBenefitArray],
    wtp_thresholds: Union[np.ndarray, List[float]],
    strategy_names: Optional[List[str]] = None,
    xlabel: str = "Willingness-to-Pay Threshold",
    ylabel: str = "Probability Cost-Effective",
    title: str = "Cost-Effectiveness Acceptability Curve (CEAC)",
    ax: Optional[Axes] = None,
    **plot_kwargs_per_strategy: Optional[
        List[dict]
    ],  # List of dicts for each strategy's plot call
) -> Axes:
    """Plot a Cost-Effectiveness Acceptability Curve (CEAC).

    A CEAC shows the probability that each strategy is optimal (has the highest
    net benefit) across a range of willingness-to-pay (WTP) thresholds.

    Args:
        nb_array (Union[np.ndarray, NetBenefitArray]):
            Net Benefit Array.
            If np.ndarray: A 3D array (n_samples, n_strategies, n_wtp_thresholds) of net benefits.
                           Or, if nb_array is (n_samples, n_strategies) and wtp_thresholds are used
                           to calculate NMB on the fly (this is less direct for CEAC).
                           This function expects NMBs already calculated for each WTP.
            If NetBenefitArray: `values` attribute should be 3D as above.
                                (This might need refinement of NetBenefitArray structure or this function)
            For simplicity, let's assume nb_array's last dimension corresponds to wtp_thresholds.
        wtp_thresholds (Union[np.ndarray, List[float]]):
            Array or list of WTP thresholds. The length must match the size of
            the last dimension of `nb_array.values` if it's 3D.
        strategy_names (Optional[List[str]]):
            Names for each strategy. If None, generic names will be used.
            Length must match `nb_array.shape[1]`.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ax (Optional[Axes]): Matplotlib Axes object to plot on.
        **plot_kwargs_per_strategy: A list of dictionaries, where each dictionary contains
                                     kwargs for the `ax.plot()` call for the corresponding strategy.
                                     If not provided, default styling is used.

    Returns
    -------
        Axes: The Matplotlib Axes object with the plot.

    Raises
    ------
        PlottingError: If Matplotlib is not installed.
        InputError: If input dimensions or lengths are mismatched.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise PlottingError(
            "Matplotlib is required for plotting functions but not installed."
        )

    if isinstance(nb_array, NetBenefitArray):
        nb_values = nb_array.values  # Expected (n_samples, n_strategies, n_wtp)
        if strategy_names is None:  # Try to get from NetBenefitArray if available
            strategy_names = nb_array.strategy_names
    elif isinstance(nb_array, np.ndarray):
        nb_values = nb_array
    else:
        raise InputError("nb_array must be a NumPy array or NetBenefitArray instance.")

    if nb_values.ndim != 3:
        raise InputError(
            "For CEAC, nb_values must be a 3D array (samples x strategies x WTP thresholds)."
            "Ensure net benefits are calculated for each WTP.",
        )

    n_samples, n_strategies, n_wtp_points = nb_values.shape
    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)

    if len(wtp_arr) != n_wtp_points:
        raise InputError(
            f"Length of wtp_thresholds ({len(wtp_arr)}) must match the third dimension "
            f"of nb_values ({n_wtp_points}).",
        )

    if strategy_names is None:
        strategy_names = [f"Strategy {i + 1}" for i in range(n_strategies)]
    elif len(strategy_names) != n_strategies:
        raise InputError(
            f"Length of strategy_names ({len(strategy_names)}) must match the second dimension "
            f"of nb_values ({n_strategies}).",
        )

    if ax is None:
        fig, ax = plt.subplots()  # type: ignore

    prob_ce = _calculate_prob_ce(nb_values, n_strategies, n_wtp_points, n_samples)

    # Plot CEAC for each strategy
    user_plot_kwargs = {}
    if plot_kwargs_per_strategy is not None:
        if (
            isinstance(plot_kwargs_per_strategy, list)
            and len(plot_kwargs_per_strategy) == n_strategies
        ):
            user_plot_kwargs = {i: kw for i, kw in enumerate(plot_kwargs_per_strategy)}
        else:
            # print("Warning: plot_kwargs_per_strategy should be a list of dicts, one for each strategy. Using defaults.")
            pass

    for s_idx in range(n_strategies):
        current_kwargs = {"label": strategy_names[s_idx]}
        # Apply default cmap colors if no specific color is given
        if "color" not in user_plot_kwargs.get(s_idx, {}):
            # Cycle through default matplotlib colors
            prop_cycle = plt.rcParams["axes.prop_cycle"]  # type: ignore
            current_kwargs["color"] = prop_cycle.by_key()["color"][
                s_idx % len(prop_cycle.by_key()["color"])
            ]

        current_kwargs.update(
            user_plot_kwargs.get(s_idx, {})
        )  # Override with user specifics
        ax.plot(wtp_arr, prob_ce[s_idx, :], **current_kwargs)  # type: ignore

    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.legend(loc="best")  # type: ignore
    ax.grid(True, linestyle=":", alpha=0.7)  # type: ignore
    ax.set_ylim(0, 1.05)  # Probability from 0 to 1

    return ax  # type: ignore


def plot_ce_plane(
    delta_costs: Union[np.ndarray, List[float]],  # Incremental costs vs comparator
    delta_effects: Union[np.ndarray, List[float]],  # Incremental effects vs comparator
    wtp_threshold: Optional[float] = None,  # WTP to draw the threshold line
    comparator_name: str = "Comparator",
    intervention_name: str = "Intervention",
    xlabel: str = "Incremental Effects (e.g., QALYs)",
    ylabel: str = "Incremental Costs",
    title: str = "Cost-Effectiveness Plane",
    ax: Optional[Axes] = None,
    scatter_kwargs: Optional[dict] = None,
    wtp_line_kwargs: Optional[dict] = None,
) -> Axes:
    """Plot a Cost-Effectiveness Plane.

    This shows a scatter plot of incremental costs vs. incremental effects for an
    intervention compared to a comparator. A WTP threshold line can be added.

    Args:
        delta_costs (Union[np.ndarray, List[float]]):
            1D array of incremental costs (Intervention - Comparator).
        delta_effects (Union[np.ndarray, List[float]]):
            1D array of incremental effects. Must be same length as delta_costs.
        wtp_threshold (Optional[float]):
            If provided, a WTP threshold line (Cost = WTP * Effect) is drawn.
        comparator_name (str): Name of the comparator for legend/annotations.
        intervention_name (str): Name of the intervention.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        ax (Optional[Axes]): Matplotlib Axes to plot on.
        scatter_kwargs (Optional[dict]): Kwargs for the scatter plot points.
        wtp_line_kwargs (Optional[dict]): Kwargs for the WTP threshold line.

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

    d_costs = np.asarray(delta_costs, dtype=DEFAULT_DTYPE)
    d_effects = np.asarray(delta_effects, dtype=DEFAULT_DTYPE)

    if d_costs.ndim != 1 or d_effects.ndim != 1:
        raise InputError("delta_costs and delta_effects must be 1-dimensional.")
    if len(d_costs) != len(d_effects):
        raise InputError("Length of delta_costs must match length of delta_effects.")

    _scatter_kwargs = {"alpha": 0.5, "s": 10}  # s is size
    if scatter_kwargs:
        _scatter_kwargs.update(scatter_kwargs)

    _wtp_line_kwargs = {
        "color": "red",
        "linestyle": "--",
        "label": f"WTP = {wtp_threshold}",
    }
    if wtp_line_kwargs:
        _wtp_line_kwargs.update(wtp_line_kwargs)

    if ax is None:
        fig, ax = plt.subplots()  # type: ignore

    ax.scatter(d_effects, d_costs, **_scatter_kwargs)  # type: ignore
    ax.axhline(0, color="black", linewidth=0.5)  # type: ignore
    ax.axvline(0, color="black", linewidth=0.5)  # type: ignore

    if wtp_threshold is not None:
        if not isinstance(wtp_threshold, (int, float)) or wtp_threshold < 0:
            # print("Warning: Invalid wtp_threshold provided for CE plane line. Skipping line.")
            pass
        else:
            # Determine line endpoints based on data range
            x_lim = ax.get_xlim()  # type: ignore
            # Use effect limits to draw the WTP line across the plot
            line_effects = np.array(x_lim)
            line_costs = wtp_threshold * line_effects
            # Update label if it was default and wtp_threshold changed
            if "label" not in _wtp_line_kwargs or _wtp_line_kwargs["label"].startswith(
                "WTP = "
            ):
                _wtp_line_kwargs["label"] = f"WTP = {wtp_threshold:,.0f}"

            ax.plot(line_effects, line_costs, **_wtp_line_kwargs)  # type: ignore
            ax.legend(loc="best")  # type: ignore

    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.grid(True, linestyle=":", alpha=0.7)  # type: ignore

    return ax  # type: ignore


# Future plots:
# - plot_evppi_surface (for 2 parameters of interest)
# - plot_enb_distribution (histogram/density of ENB for strategies)


if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed. Skipping plot generation examples.")
    else:
        print("--- Testing ceac.py ---")
        np.random.seed(0)
        n_s, n_strat, n_wtp = 1000, 3, 50

        # Dummy Net Benefit Array (samples, strategies, WTPs)
        # Strategy 1: NB increases with WTP
        # Strategy 2: NB constant
        # Strategy 3: NB decreases with WTP (less cost-effective at high WTPs)
        wtp_test_ceac = np.linspace(0, 100000, n_wtp, dtype=DEFAULT_DTYPE)

        nb_test_data = np.zeros((n_s, n_strat, n_wtp), dtype=DEFAULT_DTYPE)
        # Base NB for each strategy (randomness per sample)
        base_nb_s1 = np.random.normal(5000, 2000, n_s)
        base_nb_s2 = np.random.normal(15000, 3000, n_s)
        base_nb_s3 = np.random.normal(25000, 2500, n_s)

        for w_idx, wtp_val in enumerate(wtp_test_ceac):
            nb_test_data[:, 0, w_idx] = (
                base_nb_s1 + 0.1 * wtp_val
            )  # Strat 1 benefits from WTP
            nb_test_data[:, 1, w_idx] = base_nb_s2  # Strat 2 is flat with WTP
            nb_test_data[:, 2, w_idx] = (
                base_nb_s3 - 0.05 * wtp_val
            )  # Strat 3 becomes worse

        strat_names_test = ["Treatment A", "Treatment B (Standard)", "Treatment C"]

        fig_ceac, ax_ceac = plt.subplots()  # type: ignore
        plot_ceac(
            nb_test_data, wtp_test_ceac, strategy_names=strat_names_test, ax=ax_ceac
        )
        # plt.show() # Uncomment to display
        print("plot_ceac example generated.")
        plt.close(fig_ceac)  # type: ignore

        # Test plot_ce_plane
        n_samples_cep = 500
        delta_qaly_test = np.random.normal(0.5, 0.3, n_samples_cep)  # Incremental QALYs
        delta_cost_test = np.random.normal(
            10000, 5000, n_samples_cep
        )  # Incremental Costs

        fig_cep, ax_cep = plt.subplots()  # type: ignore
        plot_ce_plane(
            delta_costs=delta_cost_test,
            delta_effects=delta_qaly_test,
            wtp_threshold=30000,
            ax=ax_cep,
        )
        # plt.show() # Uncomment to display
        print("plot_ce_plane example generated.")
        plt.close(fig_cep)  # type: ignore

        print(
            "--- ceac.py tests completed (plots generated if Matplotlib available) ---"
        )
