"""Implementation-adjusted VOI calculations."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray


@dataclass(frozen=True)
class ImplementationAdjustedResult:
    """Structured implementation-adjusted VOI result."""

    value: float
    baseline_expected_net_benefits: np.ndarray
    baseline_optimal_strategy_index: int
    baseline_optimal_strategy_name: str
    adjusted_expected_net_benefits: np.ndarray
    adjusted_optimal_strategy_index: int
    adjusted_optimal_strategy_name: str
    implementation_multiplier: float
    uptake: float
    adherence: float
    coverage: float
    implementation_delay: float
    implementation_uncertainty: float
    discount_rate: float
    time_horizon: float | None
    population: float | None
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def _coerce_value_array(value_array: ValueArray) -> np.ndarray:
    """Validate and coerce the net-benefit matrix."""
    if not isinstance(value_array, ValueArray):
        raise_input_error("`value_array` must be a ValueArray object.")

    nb_values = np.asarray(value_array.numpy_values, dtype=DEFAULT_DTYPE)
    if nb_values.ndim != 2:
        raise_input_error(
            "Implementation-adjusted VOI requires 2D net benefits "
            "(samples x strategies)."
        )
    if nb_values.shape[0] < 1 or nb_values.shape[1] < 1:
        raise_input_error("At least one sample and one strategy are required.")
    if not np.all(np.isfinite(nb_values)):
        raise_input_error("Net-benefit values must be finite.")
    return nb_values


def _validate_probability_like(name: str, value: float) -> float:
    """Validate a probability-like input in the inclusive range [0, 1]."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise_input_error(f"`{name}` must be finite.")
    if numeric < 0.0 or numeric > 1.0:
        raise_input_error(f"`{name}` must be between 0 and 1.")
    return numeric


def _validate_positive(name: str, value: float) -> float:
    """Validate a strictly positive numeric input."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise_input_error(f"`{name}` must be positive.")
    return numeric


def value_of_implementation(
    value_array: ValueArray,
    uptake: float = 1.0,
    adherence: float = 1.0,
    coverage: float = 1.0,
    implementation_delay: float = 0.0,
    implementation_uncertainty: float = 0.0,
    discount_rate: float = 0.0,
    time_horizon: float | None = None,
    population: float | None = None,
    strategy_names: list[str] | None = None,
) -> ImplementationAdjustedResult:
    r"""Calculate the value of implementation-adjusted adoption.

    Parameters
    ----------
    value_array : ValueArray
        2D net-benefit samples with shape ``(n_samples, n_strategies)``.
    uptake, adherence, coverage : float, default=1.0
        Implementation fraction inputs in the range ``[0, 1]``.
    implementation_delay : float, default=0.0
        Delay before the intervention is implemented, in years.
    implementation_uncertainty : float, default=0.0
        Additional uncertainty penalty in the range ``[0, 1]``.
    discount_rate : float, default=0.0
        Annual discount rate used to discount delayed implementation.
    time_horizon : float, optional
        Optional time horizon for reporting and future scaling.
    population : float, optional
        Optional population size for population-adjusted reporting.
    strategy_names : list[str], optional
        Optional strategy labels.

    Returns
    -------
    ImplementationAdjustedResult
        Baseline and implementation-adjusted summaries.

    Notes
    -----
    The implementation multiplier combines uptake, adherence, coverage,
    uncertainty, and delay:

    .. math::

       m = u \times a \times c \times (1 - \delta) \times (1 + r)^{-t}.

    The reported value is the non-negative difference between the baseline
    optimal expected net benefit and the adjusted optimal expected net
    benefit, optionally scaled by population and the discounted time horizon.

    References
    ----------
    Claxton, K., Sculpher, M., & Palmer, S. (2011). Considerations for
    modelling implementation and uptake in economic evaluation.
    Phelps, C., & Mushlin, A. (1991). Focusing technology assessment using
    medical decision analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.implementation import value_of_implementation
    >>> from voiage.schema import ValueArray
    >>> values = np.array([[10.0, 12.0], [11.0, 11.5]])
    >>> result = value_of_implementation(ValueArray.from_numpy(values, ["A", "B"]))
    >>> result.value >= 0.0
    True
    """
    nb_values = _coerce_value_array(value_array)
    uptake_value = _validate_probability_like("uptake", uptake)
    adherence_value = _validate_probability_like("adherence", adherence)
    coverage_value = _validate_probability_like("coverage", coverage)
    uncertainty_value = _validate_probability_like(
        "implementation_uncertainty", implementation_uncertainty
    )
    delay_value = float(implementation_delay)
    if not np.isfinite(delay_value) or delay_value < 0.0:
        raise_input_error("`implementation_delay` must be non-negative.")
    discount_value = float(discount_rate)
    if not np.isfinite(discount_value) or discount_value < 0.0:
        raise_input_error("`discount_rate` must be non-negative.")

    if time_horizon is not None:
        time_horizon = _validate_positive("time_horizon", time_horizon)
    if population is not None:
        population = _validate_positive("population", population)

    final_strategy_names = (
        list(strategy_names)
        if strategy_names is not None
        else value_array.strategy_names
    )
    if len(final_strategy_names) != nb_values.shape[1]:
        raise_input_error(
            "`strategy_names` length must match the number of strategies."
        )

    baseline_expected_net_benefits = np.mean(nb_values, axis=0)
    baseline_optimal_strategy_index = int(np.argmax(baseline_expected_net_benefits))
    baseline_optimal_strategy_name = final_strategy_names[
        baseline_optimal_strategy_index
    ]

    implementation_multiplier = (
        uptake_value
        * adherence_value
        * coverage_value
        * max(0.0, 1.0 - uncertainty_value)
        * (1.0 / (1.0 + discount_value) ** delay_value if delay_value > 0 else 1.0)
    )
    adjusted_expected_net_benefits = (
        baseline_expected_net_benefits * implementation_multiplier
    )
    adjusted_optimal_strategy_index = int(np.argmax(adjusted_expected_net_benefits))
    adjusted_optimal_strategy_name = final_strategy_names[
        adjusted_optimal_strategy_index
    ]

    baseline_value = float(
        baseline_expected_net_benefits[baseline_optimal_strategy_index]
    )
    adjusted_value = float(
        adjusted_expected_net_benefits[adjusted_optimal_strategy_index]
    )
    value = max(0.0, baseline_value - adjusted_value)

    if population is not None and time_horizon is not None:
        annuity = (
            time_horizon
            if discount_value == 0.0
            else (1.0 - (1.0 + discount_value) ** (-time_horizon)) / discount_value
        )
        value *= population * annuity

    return ImplementationAdjustedResult(
        value=value,
        baseline_expected_net_benefits=baseline_expected_net_benefits,
        baseline_optimal_strategy_index=baseline_optimal_strategy_index,
        baseline_optimal_strategy_name=baseline_optimal_strategy_name,
        adjusted_expected_net_benefits=adjusted_expected_net_benefits,
        adjusted_optimal_strategy_index=adjusted_optimal_strategy_index,
        adjusted_optimal_strategy_name=adjusted_optimal_strategy_name,
        implementation_multiplier=float(implementation_multiplier),
        uptake=uptake_value,
        adherence=adherence_value,
        coverage=coverage_value,
        implementation_delay=delay_value,
        implementation_uncertainty=uncertainty_value,
        discount_rate=discount_value,
        time_horizon=time_horizon,
        population=population,
        method_maturity="experimental",
        diagnostics={
            "n_samples": int(nb_values.shape[0]),
            "n_strategies": int(nb_values.shape[1]),
            "baseline_value": baseline_value,
            "adjusted_value": adjusted_value,
        },
        reporting=build_cheers_reporting(
            analysis_type="value_of_implementation",
            method_family="value_of_implementation",
            method_maturity="experimental",
            population=population,
            estimator="deterministic_multiplier",
            diagnostics={
                "n_samples": int(nb_values.shape[0]),
                "n_strategies": int(nb_values.shape[1]),
                "baseline_value": baseline_value,
                "adjusted_value": adjusted_value,
            },
        ),
    )
