# voiage/methods/basic.py

"""Implementation of basic Value of Information methods.

- EVPI (Expected Value of Perfect Information)
- EVPPI (Expected Value of Partial Perfect Information)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
import warnings

import numpy as np

from voiage import _runtime
from voiage.analysis import RegressionModelProtocol
from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import (
    raise_dimension_mismatch_error,
    raise_input_error,
)
from voiage.schema import ParameterSet as PSASample

if TYPE_CHECKING:
    from voiage.schema import ValueArray

SKLEARN_AVAILABLE = False
LinearRegression = None
try:
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression

    LinearRegression = SklearnLinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    # sklearn is an optional dependency, only required for EVPPI.
    # Users will be warned if they try to use EVPPI without it.
    pass


@dataclass(frozen=True)
class ExpectedLossResult:
    """Expected opportunity-loss summary for a decision problem.

    Attributes
    ----------
    expected_net_benefit_by_strategy : numpy.ndarray
        Mean net or generalized benefit for each strategy.
    expected_opportunity_loss_by_strategy : numpy.ndarray
        Mean sample-level regret for each strategy.
    optimal_strategy_index : int
        Lowest-index strategy with the greatest expected benefit.
    minimum_expected_opportunity_loss : float
        Expected loss of the current-information optimum; equal to EVPI for
        the same sample matrix up to floating-point rounding.
    sample_count : int
        Number of uncertainty samples.
    strategy_count : int
        Number of decision strategies.
    strategy_names : list[str]
        Ordered strategy labels aligned with the result arrays.
    """

    expected_net_benefit_by_strategy: np.ndarray
    expected_opportunity_loss_by_strategy: np.ndarray
    optimal_strategy_index: int
    minimum_expected_opportunity_loss: float
    sample_count: int
    strategy_count: int
    strategy_names: list[str]

    def to_dict(
        self,
        *,
        analysis_id: str,
        decision_problem_id: str,
    ) -> dict[str, object]:
        """Serialize this result to the canonical expected-loss payload."""
        return _runtime.serialize_expected_loss_result(
            analysis_id=analysis_id,
            decision_problem_id=decision_problem_id,
            strategy_names=list(self.strategy_names),
            expected_net_benefit_by_strategy=(
                self.expected_net_benefit_by_strategy.tolist()
            ),
            expected_opportunity_loss_by_strategy=(
                self.expected_opportunity_loss_by_strategy.tolist()
            ),
            optimal_strategy_index=self.optimal_strategy_index,
            minimum_expected_opportunity_loss=self.minimum_expected_opportunity_loss,
            sample_count=self.sample_count,
            method="expected-opportunity-loss",
            reporting=None,
        )


def expected_loss(
    nb_array: Union[np.ndarray, "ValueArray"],
) -> ExpectedLossResult:
    r"""Calculate expected opportunity loss for every strategy.

    Parameters
    ----------
    nb_array : numpy.ndarray or ValueArray
        Net- or generalized-benefit samples with shape
        ``(n_samples, n_strategies)``.

    Returns
    -------
    ExpectedLossResult
        Expected benefits and opportunity losses, together with the
        current-information optimum.

    Notes
    -----
    For strategy :math:`d` and uncertain state :math:`\theta`, opportunity
    loss is

    .. math::

       L_d(\theta) = \max_j NB_j(\theta) - NB_d(\theta).

    The expected loss of the strategy with greatest expected net benefit is
    equal to EVPI. Exact ties in expected net benefit resolve to the lowest
    strategy index.
    """
    raw_values = (
        nb_array.numpy_values if hasattr(nb_array, "numpy_values") else nb_array
    )
    strategy_names = (
        list(nb_array.strategy_names) if hasattr(nb_array, "strategy_names") else []
    )
    try:
        values = np.asarray(raw_values, dtype=DEFAULT_DTYPE)
    except (TypeError, ValueError) as exc:
        raise_input_error(f"Net-benefit values must be numeric: {exc}")
    payload = _runtime.compute_expected_loss(values.tolist())
    if not strategy_names:
        strategy_names = [
            f"Strategy {index + 1}" for index in range(int(payload["strategy_count"]))
        ]
    return ExpectedLossResult(
        expected_net_benefit_by_strategy=np.asarray(
            payload["expected_net_benefit_by_strategy"],
            dtype=DEFAULT_DTYPE,
        ),
        expected_opportunity_loss_by_strategy=np.asarray(
            payload["expected_opportunity_loss_by_strategy"],
            dtype=DEFAULT_DTYPE,
        ),
        optimal_strategy_index=int(payload["optimal_strategy_index"]),
        minimum_expected_opportunity_loss=float(
            payload["minimum_expected_opportunity_loss"]
        ),
        sample_count=int(payload["sample_count"]),
        strategy_count=int(payload["strategy_count"]),
        strategy_names=strategy_names,
    )


def check_parameter_samples(parameter_samples: object, n_samples: int) -> np.ndarray:
    """Validate and normalize parameter samples for EVPPI.

    Parameters
    ----------
    parameter_samples : object
        Parameter samples supplied as a NumPy array, a ``PSASample``, or a
        mapping from parameter names to arrays.
    n_samples : int
        Expected sample count from the associated net-benefit array.

    Returns
    -------
    numpy.ndarray
        A 2D array with shape ``(n_samples, n_parameters)``.

    Raises
    ------
    InputError
        If the input type is unsupported or the sample counts do not match.

    Notes
    -----
    This helper is intentionally strict: EVPPI requires aligned PSA samples so
    the parameter matrix and net-benefit matrix can be paired sample by sample.
    """
    if isinstance(parameter_samples, np.ndarray):
        x = parameter_samples
    elif isinstance(parameter_samples, PSASample):
        if isinstance(parameter_samples.parameters, dict):
            x = np.stack(list(parameter_samples.parameters.values()), axis=1)
        else:
            # Handle xarray or other types if necessary
            raise_input_error(
                "PSASample with non-dict parameters not yet supported for EVPPI."
            )
    elif isinstance(parameter_samples, dict):
        x = np.stack(list(parameter_samples.values()), axis=1)
    else:
        raise_input_error(
            f"`parameter_samples` must be a NumPy array, PSASample, or Dict. Got {type(parameter_samples)}."
        )

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.shape[0] != n_samples:
        raise_dimension_mismatch_error(
            f"Number of samples in `parameter_samples` ({x.shape[0]}) "
            f"does not match `nb_array` ({n_samples})."
        )
    return x


def evpi(
    nb_array: Union[np.ndarray, "ValueArray"],
    population: float | None = None,
    time_horizon: float | None = None,
    discount_rate: float | None = None,
) -> float:
    r"""Calculate expected value of perfect information.

    Parameters
    ----------
    nb_array : numpy.ndarray or ValueArray
        Net-benefit samples with shape ``(n_samples, n_strategies)``.
    population : float, optional
        Population size for population-scaled EVPI.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.

    Returns
    -------
    float
        EVPI on a per-decision basis unless population scaling is requested.

    Notes
    -----
    EVPI is the gap between the expected value of the strategy that would be
    chosen with perfect information and the value of the strategy chosen under
    current information:

    .. math::

       \mathrm{EVPI} = E\left[\max_i NB_i(\theta)\right] - \max_i E[NB_i(\theta)].

    References
    ----------
    Briggs, A. H., Claxton, K., & Sculpher, M. J. (2006). *Decision Modelling
    for Health Economic Evaluation*. Oxford University Press.
    Claxton, K. (1999). The irrelevance of inference: A decision-making
    approach to the stochastic evaluation of health care technologies.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.basic import evpi
    >>> nb = np.array([[10.0, 12.0], [11.0, 9.0], [13.0, 14.0]])
    >>> round(evpi(nb), 6)
    0.666667
    """
    from voiage.analysis import DecisionAnalysis  # Local import to avoid circularity

    try:
        values = (
            nb_array.numpy_values
            if hasattr(nb_array, "numpy_values")
            else np.asarray(nb_array, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise_input_error(f"Net-benefit values must be numeric: {exc}")
    if not np.all(np.isfinite(values)):
        raise_input_error(
            "Net-benefit values must be finite.",
            diagnostic_code="non_finite_value",
        )
    analysis = DecisionAnalysis(nb_array=nb_array)
    return analysis.evpi(
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )


def evppi(
    nb_array: Union[np.ndarray, "ValueArray"],
    parameter_samples: Union[np.ndarray, "PSASample", dict[str, np.ndarray]],
    parameters_of_interest: list[str],
    population: float | None = None,
    time_horizon: float | None = None,
    discount_rate: float | None = None,
    n_regression_samples: int | None = None,
    chunk_size: int | None = None,
    regression_model: RegressionModelProtocol
    | type[RegressionModelProtocol]
    | None = None,
) -> float:
    r"""Calculate expected value of partial perfect information.

    Parameters
    ----------
    nb_array : numpy.ndarray or ValueArray
        Net-benefit samples with shape ``(n_samples, n_strategies)``.
    parameter_samples : numpy.ndarray, PSASample, or dict[str, numpy.ndarray]
        PSA parameter samples aligned to ``nb_array``.
    parameters_of_interest : list[str]
        Parameter names to retain for the EVPPI calculation.
    population : float, optional
        Population size for population-scaled EVPPI.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.
    n_regression_samples : int, optional
        Number of samples to use in the regression approximation.
    chunk_size : int, optional
        Optional batch size for chunked evaluation.
    regression_model : RegressionModelProtocol or type, optional
        Optional regression model used by the analysis layer.

    Returns
    -------
    float
        EVPPI on a per-decision basis unless population scaling is requested.

    Notes
    -----
    EVPPI measures the gain from resolving a subset of uncertain parameters
    while leaving the remainder uncertain:

    .. math::

       \mathrm{EVPPI}(x) = E_x\left[\max_d E[NB_d \mid x]\right] - \max_d E[NB_d].

    The implementation delegates approximation details to
    :class:`~voiage.analysis.DecisionAnalysis`, so the precise estimator can
    vary with the chosen regression model or sample controls.

    References
    ----------
    Strong, M., Oakley, J. E., & Brennan, A. (2014). Estimating multiparameter
    partial expected value of perfect information from the posterior
    distribution. *Medical Decision Making*, 34(3), 314-326.
    Ades, A. E., Lu, G., & Claxton, K. (2004). Expected value of sample
    information calculations in medical decision modeling.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.basic import evppi
    >>> from voiage.schema import ParameterSet
    >>> nb = np.array([[10.0, 12.0], [11.0, 9.0], [13.0, 14.0]])
    >>> params = ParameterSet.from_numpy_or_dict({
    ...     "effect": np.array([0.1, 0.2, 0.3]),
    ...     "cost": np.array([1.0, 1.1, 0.9]),
    ... })
    >>> round(evppi(nb, params, ["effect"]), 6) >= 0.0
    True
    """
    from voiage.analysis import DecisionAnalysis  # Local import to avoid circularity

    try:
        values = (
            nb_array.numpy_values
            if hasattr(nb_array, "numpy_values")
            else np.asarray(nb_array, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise_input_error(f"Net-benefit values must be numeric: {exc}")
    if not np.all(np.isfinite(values)):
        raise_input_error(
            "Net-benefit values must be finite.",
            diagnostic_code="non_finite_value",
        )
    parameter_values = (
        parameter_samples.parameters.values()
        if isinstance(parameter_samples, PSASample)
        else parameter_samples.values()
        if isinstance(parameter_samples, dict)
        else (parameter_samples,)
    )
    try:
        parameters_are_finite = all(
            np.all(np.isfinite(np.asarray(item, dtype=float)))
            for item in parameter_values
        )
    except (TypeError, ValueError) as exc:
        raise_input_error(f"Parameter samples must be numeric: {exc}")
    if not parameters_are_finite:
        raise_input_error(
            "Parameter samples must be finite.",
            diagnostic_code="non_finite_value",
        )

    filtered_parameter_samples: np.ndarray | PSASample | dict[str, np.ndarray]

    if isinstance(parameter_samples, dict):
        warnings.warn(
            "[deprecated_raw_dict_parameter_samples] Passing raw dict parameter "
            "samples to `voiage.methods.basic.evppi` "
            "is a compatibility alias. Pass a `ParameterSet` instead.",
            FutureWarning,
            stacklevel=2,
        )

    # Filter parameter samples to only include parameters of interest
    if isinstance(parameter_samples, PSASample):
        # Check that all parameters of interest are in the parameter set
        available_params = set(parameter_samples.parameter_names)
        requested_params = set(parameters_of_interest)
        missing_params = requested_params - available_params
        if missing_params:
            raise_input_error(
                f"All `parameters_of_interest` must be in the ParameterSet. Missing: {missing_params}"
            )

        # Create a new ParameterSet with only the parameters of interest
        filtered_parameters = {
            name: parameter_samples.parameters[name] for name in parameters_of_interest
        }
        import xarray as xr

        dataset = xr.Dataset(
            {k: ("n_samples", v) for k, v in filtered_parameters.items()},
            coords={
                "n_samples": np.arange(len(next(iter(filtered_parameters.values()))))
            },
        )
        from voiage.schema import ParameterSet

        filtered_parameter_samples = ParameterSet(dataset=dataset)
    elif isinstance(parameter_samples, dict):
        # Check that all parameters of interest are in the parameter dict
        available_params = set(parameter_samples.keys())
        requested_params = set(parameters_of_interest)
        missing_params = requested_params - available_params
        if missing_params:
            raise_input_error(
                f"All `parameters_of_interest` must be in the ParameterSet. Missing: {missing_params}"
            )

        # Filter the dictionary
        filtered_parameter_samples = {
            name: parameter_samples[name] for name in parameters_of_interest
        }
    else:
        # For numpy arrays, we assume the caller has already filtered the parameters
        filtered_parameter_samples = parameter_samples

    analysis = DecisionAnalysis(
        nb_array=nb_array, parameter_samples=filtered_parameter_samples
    )
    return analysis.evppi(
        parameters_of_interest=parameters_of_interest,
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
        n_regression_samples=n_regression_samples,
        chunk_size=chunk_size,
        regression_model=regression_model,
    )
