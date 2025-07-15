import matplotlib.pyplot as plt
import numpy as np

from voiage.schema import ValueArray


def plot_ceac(
    value_array: ValueArray,
    wtp_range: np.ndarray,
    ax=None,
    show=True,
    **kwargs,
):
    """
    Plot the Cost-Effectiveness Acceptability Curve (CEAC).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ceac = np.zeros((len(wtp_range), value_array.n_strategies))
    for i, wtp in enumerate(wtp_range):
        nmb = value_array.values * wtp
        optimal_strategy = np.argmax(nmb, axis=1)
        for j in range(value_array.n_strategies):
            ceac[i, j] = np.mean(optimal_strategy == j)

    for j in range(value_array.n_strategies):
        ax.plot(wtp_range, ceac[:, j], label=value_array.strategy_names[j], **kwargs)

    ax.set_xlabel("Willingness-to-Pay Threshold")
    ax.set_ylabel("Probability of Being Cost-Effective")
    ax.legend()

    if show:
        plt.show()

    return fig, ax


from typing import List, Optional, Union
from matplotlib.axes import Axes


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
