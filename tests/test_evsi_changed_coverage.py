"""Coverage-focused tests for the v2 EVSI scientific contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import xarray as xr

from voiage import _runtime
from voiage.exceptions import InputError
from voiage.methods import sample_information
from voiage.methods.sample_information import evsi, normal_normal_two_arm_evsi
from voiage.schema import DecisionOption, ParameterSet, TrialDesign, ValueArray

if TYPE_CHECKING:
    from collections.abc import Callable


def _prior() -> ParameterSet:
    return ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.linspace(-0.2, 0.2, 12),
            "mean_programme": np.linspace(-0.1, 0.3, 12),
            "sd_outcome": np.ones(12),
        }
    )


def _design() -> TrialDesign:
    return TrialDesign([DecisionOption("Control", 10), DecisionOption("Programme", 10)])


def _model(parameters: ParameterSet) -> ValueArray:
    effect = (
        parameters.parameters["mean_programme"] - parameters.parameters["mean_control"]
    )
    return ValueArray.from_numpy(
        np.column_stack([np.zeros(parameters.n_samples), effect]),
        ["Control", "Programme"],
    )


def _analytical_evsi(**overrides: object) -> float:
    arguments: dict[str, object] = {
        "prior_mean": 0.0,
        "prior_standard_deviation": 1.0,
        "outcome_standard_deviation": 1.0,
        "total_sample_size": 20,
        "net_benefit_slope": 1.0,
        "net_benefit_intercept": 0.0,
    }
    arguments.update(overrides)
    return normal_normal_two_arm_evsi(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argument", "value", "diagnostic_code"),
    [
        ("prior_mean", True, "invalid_scalar_input"),
        ("prior_standard_deviation", object(), "invalid_scalar_input"),
        ("outcome_standard_deviation", np.inf, "non_finite_value"),
    ],
)
def test_analytical_evsi_rejects_invalid_public_scalars(
    argument: str,
    value: object,
    diagnostic_code: str,
) -> None:
    with pytest.raises(InputError) as captured:
        _analytical_evsi(**{argument: value})

    assert captured.value.diagnostic_code == diagnostic_code


def test_analytical_evsi_translates_native_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingNative:
        def compute_normal_normal_two_arm_evsi(
            self,
            *_args: object,
        ) -> float:
            raise RuntimeError("forced native EVSI failure")

    monkeypatch.setattr(_runtime, "_native", FailingNative)

    with pytest.raises(RuntimeError, match="forced native EVSI failure"):
        _analytical_evsi()


def test_builtin_contract_requires_outcome_sd_and_arm_parameters() -> None:
    no_sd = ParameterSet.from_numpy_or_dict({"mean_control": np.linspace(0.0, 1.0, 4)})
    with pytest.raises(InputError, match="sd_outcome"):
        sample_information._known_outcome_standard_deviation(no_sd)

    missing_arm = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.linspace(0.0, 1.0, 4),
            "sd_outcome": np.ones(4),
        }
    )
    with pytest.raises(InputError, match="mean_programme"):
        sample_information._validate_builtin_normal_contract(missing_arm, _design())


def test_builtin_contract_rejects_non_finite_arm_sample_size() -> None:
    arm = DecisionOption("Control", 10)
    object.__setattr__(arm, "sample_size", np.inf)
    design = TrialDesign([arm])
    prior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.linspace(0.0, 1.0, 4),
            "sd_outcome": np.ones(4),
        }
    )

    with pytest.raises(InputError) as captured:
        sample_information._validate_builtin_normal_contract(prior, design)

    assert captured.value.diagnostic_code == "invalid_sample_size"


def test_outcome_sd_rejects_non_finite_squared_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_isfinite = np.isfinite

    def scalar_is_non_finite(value: object) -> Any:
        if np.isscalar(value):
            return False
        return original_isfinite(cast("Any", value))

    monkeypatch.setattr(np, "isfinite", scalar_is_non_finite)

    with pytest.raises(InputError) as captured:
        sample_information._known_outcome_standard_deviation(_prior())

    assert captured.value.diagnostic_code == "invalid_observation_scale"


def test_joint_prior_requires_uncertain_finite_scalar_draws() -> None:
    only_sd = ParameterSet.from_numpy_or_dict({"sd_outcome": np.ones(4)})
    with pytest.raises(InputError, match="uncertain model parameters"):
        sample_information._joint_normal_prior_parameters(only_sd)

    non_finite = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.array([0.0, 1.0, np.inf, 2.0]),
            "sd_outcome": np.ones(4),
        }
    )
    with pytest.raises(InputError) as captured:
        sample_information._joint_normal_prior_parameters(non_finite)
    assert captured.value.diagnostic_code == "non_finite_value"

    multidimensional = ParameterSet(
        xr.Dataset(
            {
                "mean_control": (
                    ("n_samples", "component"),
                    np.arange(8, dtype=float).reshape(4, 2),
                ),
                "sd_outcome": (("n_samples",), np.ones(4)),
            },
            coords={"n_samples": np.arange(4), "component": [0, 1]},
        )
    )
    with pytest.raises(InputError) as captured:
        sample_information._joint_normal_prior_parameters(multidimensional)
    assert captured.value.diagnostic_code == "non_scalar_parameter"


def test_joint_prior_rejects_non_finite_or_indefinite_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "cov", lambda *_args, **_kwargs: np.array([[np.nan]]))
    with pytest.raises(InputError, match="finite joint prior covariance"):
        sample_information._joint_normal_prior_parameters(_prior())

    monkeypatch.setattr(np, "cov", lambda *_args, **_kwargs: np.eye(2))
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda *_args: np.array([-1.0, 1.0]))
    with pytest.raises(InputError) as captured:
        sample_information._joint_normal_prior_parameters(_prior())
    assert captured.value.diagnostic_code == "invalid_prior_covariance"


def test_gaussian_draws_translate_decomposition_and_indefinite_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decomposition_failure(*_args: object, **_kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("forced eigensystem failure")

    monkeypatch.setattr(np.linalg, "eigh", decomposition_failure)
    with pytest.raises(InputError) as captured:
        sample_information._positive_semidefinite_draws(
            np.zeros(1),
            np.eye(1),
            2,
            np.random.default_rng(1),
        )
    assert captured.value.diagnostic_code == "covariance_decomposition_failed"

    monkeypatch.setattr(
        np.linalg,
        "eigh",
        lambda *_args: (np.array([-1.0]), np.eye(1)),
    )
    with pytest.raises(InputError) as captured:
        sample_information._positive_semidefinite_draws(
            np.zeros(1),
            np.array([[-1.0]]),
            2,
            np.random.default_rng(1),
        )
    assert captured.value.diagnostic_code == "invalid_posterior_covariance"


def test_observation_contract_rejects_non_finite_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sample_information,
        "_known_outcome_standard_deviation",
        lambda _prior: np.inf,
    )

    with pytest.raises(InputError) as captured:
        sample_information._normal_observation_contract(_prior(), _design())

    assert captured.value.diagnostic_code == "invalid_observation_scale"


def _posterior_from_contract(
    *,
    n_draws: object = 2,
    observation: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> ParameterSet:
    return sample_information._joint_normal_posterior_from_contract(
        ["mean_control"],
        np.array([0.0]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([0.0]) if observation is None else observation,
        1.0,
        cast("int", n_draws),
        np.random.default_rng(2) if rng is None else rng,
    )


@pytest.mark.parametrize("n_draws", [True, 0, 1.5])
def test_joint_posterior_rejects_invalid_draw_counts(n_draws: object) -> None:
    with pytest.raises(InputError, match="draw count"):
        _posterior_from_contract(n_draws=n_draws)


def test_joint_posterior_rejects_non_finite_observation() -> None:
    with pytest.raises(InputError, match="finite mean"):
        _posterior_from_contract(observation=np.array([np.nan]))


def test_joint_posterior_translates_predictive_linear_algebra_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cholesky_failure(*_args: object, **_kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("forced Cholesky failure")

    monkeypatch.setattr(np.linalg, "cholesky", cholesky_failure)
    with pytest.raises(InputError) as captured:
        _posterior_from_contract()
    assert captured.value.diagnostic_code == "singular_predictive_covariance"


def test_joint_posterior_normalises_draw_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sample_information,
        "_positive_semidefinite_draws",
        lambda *_args, **_kwargs: np.array([0.1, 0.2]),
    )

    posterior = _posterior_from_contract()

    np.testing.assert_allclose(posterior.parameters["mean_control"], [0.1, 0.2])


def test_bayesian_update_and_builtin_sampler_reject_invalid_trial_data() -> None:
    trial_data = {
        "Control": np.array([0.0, np.nan]),
        "Programme": np.array([0.1, 0.2]),
    }
    with pytest.raises(InputError, match="finite trial data"):
        sample_information._bayesian_update(_prior(), trial_data, _design())

    with pytest.raises(InputError, match="mapping"):
        sample_information._builtin_joint_normal_posterior_sampler(
            _prior(),
            [],
            _design(),
            2,
            np.random.default_rng(3),
        )
    with pytest.raises(InputError, match="finite mean"):
        sample_information._builtin_joint_normal_posterior_sampler(
            _prior(),
            {"Control": 0.0},
            _design(),
            2,
            np.random.default_rng(3),
        )
    with pytest.raises(InputError, match="finite mean"):
        sample_information._builtin_joint_normal_posterior_sampler(
            _prior(),
            {"Control": 0.0, "Programme": np.inf},
            _design(),
            2,
            np.random.default_rng(3),
        )


def test_builtin_trial_and_posterior_adapters_accept_valid_data() -> None:
    trial_data = sample_information._builtin_normal_trial_simulator(
        {
            "mean_control": 0.0,
            "mean_programme": 0.1,
            "sd_outcome": 1.0,
        },
        _design(),
        np.random.default_rng(4),
    )
    posterior = sample_information._builtin_joint_normal_posterior_sampler(
        _prior(),
        trial_data,
        _design(),
        3,
        np.random.default_rng(5),
    )

    assert posterior.n_samples == 3


def test_builtin_two_loop_translates_predictive_factor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cholesky_failure(*_args: object, **_kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("forced predictive failure")

    monkeypatch.setattr(np.linalg, "cholesky", cholesky_failure)
    with pytest.raises(InputError) as captured:
        sample_information._evsi_builtin_joint_normal(
            _model,
            _prior(),
            _design(),
            1,
            2,
            np.random.default_rng(6),
        )
    assert captured.value.diagnostic_code == "singular_predictive_covariance"


def test_builtin_two_loop_rejects_non_finite_posterior_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    means = iter([np.array([0.0, 0.0]), np.array([np.inf, 0.0])])
    monkeypatch.setattr(
        sample_information,
        "_finite_strategy_means",
        lambda _values: next(means),
    )

    with pytest.raises(InputError) as captured:
        sample_information._evsi_builtin_joint_normal(
            _model,
            _prior(),
            _design(),
            1,
            2,
            np.random.default_rng(7),
        )
    assert captured.value.diagnostic_code == "non_finite_reduction"


def _run_custom_two_loop(
    posterior_sampler: Callable[
        [ParameterSet, object, TrialDesign, int, np.random.Generator],
        object,
    ],
) -> float:
    return sample_information._evsi_two_loop(
        _model,
        _prior(),
        _design(),
        1,
        2,
        lambda *_args: None,
        cast("Any", posterior_sampler),
        np.random.default_rng(8),
    )


def test_custom_two_loop_validates_posterior_type_and_draw_count() -> None:
    with pytest.raises(InputError, match="must return a ParameterSet"):
        _run_custom_two_loop(lambda *_args: object())

    with pytest.raises(InputError, match="exactly"):
        _run_custom_two_loop(lambda *_args: _prior())


def test_custom_two_loop_rejects_non_finite_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = ParameterSet.from_numpy_or_dict(
        {
            "mean_control": np.array([0.0, 0.1]),
            "mean_programme": np.array([0.1, 0.2]),
            "sd_outcome": np.ones(2),
        }
    )
    monkeypatch.setattr(
        sample_information,
        "_finite_strategy_means",
        lambda _values: np.array([np.inf, 0.0]),
    )

    with pytest.raises(InputError) as captured:
        _run_custom_two_loop(lambda *_args: posterior)
    assert captured.value.diagnostic_code == "non_finite_reduction"


def test_public_two_loop_rejects_non_finite_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sample_information,
        "_evsi_builtin_joint_normal",
        lambda *_args: (np.inf, 0.0),
    )

    with pytest.raises(InputError) as captured:
        evsi(
            _model,
            _prior(),
            _design(),
            n_outer_loops=1,
            n_inner_loops=2,
        )
    assert captured.value.diagnostic_code == "non_finite_reduction"


def test_public_two_loop_warns_for_negative_builtin_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sample_information,
        "_evsi_builtin_joint_normal",
        lambda *_args: (0.5, 1.0),
    )

    with pytest.warns(RuntimeWarning, match="evsi_negative_monte_carlo_estimate"):
        result = evsi(
            _model,
            _prior(),
            _design(),
            n_outer_loops=1,
            n_inner_loops=2,
        )

    assert result == -0.5


def test_public_two_loop_rejects_non_finite_population_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sample_information,
        "_evsi_builtin_joint_normal",
        lambda *_args: (2.0, 1.0),
    )

    with pytest.raises(InputError) as captured:
        evsi(
            _model,
            _prior(),
            _design(),
            population=1e308,
            time_horizon=2.0,
            n_outer_loops=1,
            n_inner_loops=2,
        )
    assert captured.value.diagnostic_code == "non_finite_reduction"
