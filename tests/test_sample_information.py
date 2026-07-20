# tests/test_sample_information.py

"""Test VOI methods related to sample information (EVSI, ENBS)."""

import warnings

import numpy as np
import pytest

from voiage import _runtime
from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import InputError
import voiage.methods.sample_information as si_module
from voiage.schema import DecisionOption, TrialDesign, ValueArray
from voiage.schema import ParameterSet as PSASample

_bayesian_update = si_module._bayesian_update
enbs = si_module.enbs
evsi = si_module.evsi

# --- Dummy components for EVSI testing ---


def dummy_model_func_evsi(
    psa_params_or_sample: PSASample,
) -> ValueArray:
    """Define a simple model function for EVSI structure testing.

    It generates net benefits based on the number of samples in psa_params_or_sample.
    """
    n_samples = 0
    if isinstance(psa_params_or_sample, PSASample):
        n_samples = psa_params_or_sample.n_samples

    if n_samples == 0:  # Fallback if n_samples couldn't be determined
        n_samples = 3  # Default to a small number

    # For simplicity, generate random net benefits for 2 strategies
    # In a real model, these would be calculated based on input parameters
    nb_strategy1 = np.random.normal(loc=100, scale=10, size=n_samples)
    nb_strategy2 = np.random.normal(loc=105, scale=15, size=n_samples)

    # Create xarray dataset for ValueArray
    import xarray as xr

    dataset = xr.Dataset(
        {
            "net_benefit": (
                ("n_samples", "n_strategies"),
                np.stack([nb_strategy1, nb_strategy2], axis=1).astype(DEFAULT_DTYPE),
            )
        },
        coords={"n_samples": np.arange(n_samples), "n_strategies": [0, 1]},
    )
    from voiage.schema import ValueArray

    return ValueArray(dataset=dataset)


def deterministic_model_func_evsi(psa_params_or_sample: PSASample) -> ValueArray:
    """Deterministic two-strategy model for efficient EVSI tests."""
    treatment = psa_params_or_sample.parameters["mean_new_treatment"]
    standard = psa_params_or_sample.parameters["mean_standard_care"]
    unrelated = psa_params_or_sample.parameters["unrelated_param"]
    values = np.column_stack(
        [
            100.0 + 8.0 * standard,
            85.0 + 10.0 * treatment + 2.0 * unrelated,
        ]
    )
    return ValueArray.from_numpy(values, ["standard", "new"])


@pytest.fixture
def dummy_psa_for_evsi() -> PSASample:
    """Create a dummy PSASample for EVSI tests."""
    # Parameters for a Normal-Normal conjugate update scenario
    # Means are different enough to expect some EVSI.
    # sd_outcome is relatively small to make learning more impactful.
    # n_samples for psa_prior should be reasonably large for stable metamodel fitting.
    n_psa_samples = 500
    import xarray as xr

    dataset = xr.Dataset(
        {
            "mean_new_treatment": (
                ("n_samples",),
                np.random.normal(loc=10, scale=2, size=n_psa_samples).astype(
                    DEFAULT_DTYPE
                ),
            ),
            "mean_standard_care": (
                ("n_samples",),
                np.random.normal(loc=8, scale=2, size=n_psa_samples).astype(
                    DEFAULT_DTYPE
                ),
            ),
            "sd_outcome": (
                ("n_samples",),
                np.random.uniform(low=0.5, high=1.5, size=n_psa_samples).astype(
                    DEFAULT_DTYPE
                ),
            ),
            # Add another dummy parameter not directly used in update, to test metamodel with multiple params
            "unrelated_param": (
                ("n_samples",),
                np.random.rand(n_psa_samples).astype(DEFAULT_DTYPE),
            ),
        },
        coords={"n_samples": np.arange(n_psa_samples)},
    )
    from voiage.schema import ParameterSet

    return ParameterSet(dataset=dataset)


@pytest.fixture
def dummy_trial_design_for_evsi() -> TrialDesign:
    """Create a dummy TrialDesign for EVSI tests."""
    # Arm names match keys expected by _simulate_trial_data (via convention)
    # and _bayesian_update (hardcoded 'New Treatment' for data_key_for_update)
    from voiage.schema import DecisionOption, TrialDesign

    arm1 = DecisionOption(
        name="New Treatment", sample_size=50
    )  # Reduced sample size for faster test simulation
    arm2 = DecisionOption(name="Standard Care", sample_size=50)
    return TrialDesign(arms=[arm1, arm2])


# --- Tests for EVSI ---


def test_evsi_two_loop_method(dummy_psa_for_evsi, dummy_trial_design_for_evsi) -> None:
    """Test the two-loop EVSI method."""

    evsi_val = evsi(
        model_func=dummy_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=10,
        n_inner_loops=20,
    )
    assert evsi_val >= 0, "EVSI should be non-negative."


def test_evsi_two_loop_seed_is_reproducible_and_does_not_repeat_initial_callback(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """A seeded two-loop run is reproducible and has the documented callback count."""

    callback_counts = [0, 0]

    def deterministic_counting_model(
        psa_params_or_sample: PSASample,
    ) -> ValueArray:
        callback_counts[0] += 1
        standard = psa_params_or_sample.parameters["mean_standard_care"]
        treatment = psa_params_or_sample.parameters["mean_new_treatment"]
        return ValueArray.from_numpy(
            np.column_stack([standard, treatment]), ["standard", "treatment"]
        )

    first = evsi(
        deterministic_counting_model,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=4,
        n_inner_loops=2,
        seed=42,
    )
    first_callback_count = callback_counts[0]

    def deterministic_second_model(psa_params_or_sample: PSASample) -> ValueArray:
        callback_counts[1] += 1
        standard = psa_params_or_sample.parameters["mean_standard_care"]
        treatment = psa_params_or_sample.parameters["mean_new_treatment"]
        return ValueArray.from_numpy(
            np.column_stack([standard, treatment]), ["standard", "treatment"]
        )

    second = evsi(
        deterministic_second_model,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=4,
        n_inner_loops=2,
        seed=42,
    )

    assert first == second
    assert first_callback_count == 5
    assert callback_counts[1] == 5


def test_evsi_two_loop_seed_does_not_mutate_global_rng(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """A seeded run consumes only its local NumPy generator stream."""

    def deterministic_model(psa_params_or_sample: PSASample) -> ValueArray:
        standard = psa_params_or_sample.parameters["mean_standard_care"]
        treatment = psa_params_or_sample.parameters["mean_new_treatment"]
        return ValueArray.from_numpy(
            np.column_stack([standard**2, treatment**2]), ["standard", "treatment"]
        )

    np.random.seed(8675309)
    state_before = np.random.get_state()
    evsi(
        deterministic_model,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=4,
        seed=42,
    )
    after_seeded_run = np.random.random(5)

    np.random.set_state(state_before)
    expected_after_untouched_run = np.random.random(5)
    np.testing.assert_array_equal(after_seeded_run, expected_after_untouched_run)


def test_evsi_two_loop_seed_changes_stochastic_result(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Different valid seeds select independent deterministic random streams."""

    def deterministic_model(psa_params_or_sample: PSASample) -> ValueArray:
        standard = psa_params_or_sample.parameters["mean_standard_care"]
        treatment = psa_params_or_sample.parameters["mean_new_treatment"]
        return ValueArray.from_numpy(
            np.column_stack([standard**2, treatment**2]), ["standard", "treatment"]
        )

    first = evsi(
        deterministic_model,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=8,
        seed=1,
    )
    second = evsi(
        deterministic_model,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        method="two_loop",
        n_outer_loops=8,
        seed=2,
    )

    assert first != second


@pytest.mark.parametrize("invalid_seed", [-1, 2**64, 1.5, True, "42"])
def test_evsi_two_loop_rejects_invalid_seed(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, invalid_seed
) -> None:
    """Seeded execution accepts only integer values in the uint64 range."""

    with pytest.raises(
        InputError,
        match=r"seed must be an integer between 0 and 2\*\*64 - 1",
    ):
        evsi(
            dummy_model_func_evsi,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            method="two_loop",
            n_outer_loops=1,
            seed=invalid_seed,
        )


def test_evsi_regression_method_not_implemented(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Test that the regression method for EVSI raises a NotImplementedError."""
    from voiage.exceptions import VoiageNotImplementedError

    monkeypatch.setattr(si_module, "SKLEARN_AVAILABLE", False)
    with pytest.raises(VoiageNotImplementedError):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="regression",
        )


def test_evsi_regression_method(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Test the regression-based EVSI method."""

    evsi_val = evsi(
        model_func=dummy_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="regression",
        n_outer_loops=10,
    )
    assert evsi_val >= 0, "EVSI should be non-negative."


def test_evsi_efficient_method_uses_psa_regression_without_two_loop(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Test efficient EVSI computes from the PSA sample without nested loops."""

    def fail_two_loop(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("efficient EVSI should not call the two-loop method")

    monkeypatch.setattr(si_module, "_evsi_two_loop", fail_two_loop)

    def native_result(
        net_benefit: list[list[float]],
        parameter_samples: list[list[float]],
        _trial_sample_size: int,
    ) -> dict[str, object]:
        current = float(np.max(np.mean(net_benefit, axis=0)))
        return {
            "estimator": "efficient_linear",
            "contract_version": 1,
            "expected_current_value": current,
            "expected_sample_value": current + 1.0,
            "expected_perfect_information": current + 2.0,
            "information_fraction": 0.5,
            "evsi": 1.0,
            "sample_count": len(net_benefit),
            "strategy_count": len(net_benefit[0]),
            "parameter_count": len(parameter_samples[0]),
        }

    monkeypatch.setattr(_runtime, "compute_evsi_efficient_linear", native_result)

    evsi_val = si_module.evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="efficient",
    )
    assert evsi_val >= 0


def test_evsi_efficient_linear_routes_through_native_kernel(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """The retained linear efficient estimator uses the Rust contract."""

    captured: dict[str, object] = {}

    def compute(
        net_benefit: list[list[float]],
        parameter_samples: list[list[float]],
        trial_sample_size: int,
    ) -> dict[str, object]:
        captured.update(
            net_benefit=net_benefit,
            parameter_samples=parameter_samples,
            trial_sample_size=trial_sample_size,
        )
        return {
            "estimator": "efficient_linear",
            "contract_version": 1,
            "expected_current_value": float(np.max(np.mean(net_benefit, axis=0))),
            "expected_sample_value": float(np.max(np.mean(net_benefit, axis=0))) + 1.5,
            "expected_perfect_information": float(np.max(np.mean(net_benefit, axis=0)))
            + 2.0,
            "information_fraction": 1.0 / 3.0,
            "evsi": 1.5,
            "sample_count": 500,
            "strategy_count": 2,
            "parameter_count": 4,
        }

    monkeypatch.setattr(_runtime, "compute_evsi_efficient_linear", compute)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = si_module.evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="efficient",
            metamodel="linear",
        )

    assert result == pytest.approx(1.5)
    assert captured["trial_sample_size"] == 100
    assert len(captured["net_benefit"]) == 500
    assert len(captured["parameter_samples"]) == 500


def test_evsi_efficient_linear_falls_back_when_native_extension_is_absent(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """An optional native extension must not break the Python reference path."""

    def unavailable(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ModuleNotFoundError("voiage._core")

    monkeypatch.setattr(_runtime, "compute_evsi_efficient_linear", unavailable)
    with pytest.warns(DeprecationWarning, match="efficient-linear EVSI fallback"):
        result = si_module.evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="efficient",
            metamodel="linear",
        )
    assert result >= 0.0


def test_evsi_efficient_linear_falls_back_for_rank_deficient_design(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """The Python least-squares behavior remains available for deficient designs."""

    def rank_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise InputError("efficient-linear design is rank deficient")

    monkeypatch.setattr(_runtime, "compute_evsi_efficient_linear", rank_failure)
    with pytest.warns(DeprecationWarning, match="efficient-linear EVSI fallback"):
        result = si_module.evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="efficient",
            metamodel="linear",
        )
    assert result >= 0.0


def test_evsi_efficient_linear_rejects_malformed_native_envelope(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Malformed native output fails closed at the Python boundary."""
    monkeypatch.setattr(
        _runtime,
        "compute_evsi_efficient_linear",
        lambda *_args, **_kwargs: {"evsi": 1.0},
    )
    with pytest.raises(InputError, match="invalid result envelope"):
        si_module.evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="efficient",
            metamodel="linear",
        )


def test_evsi_efficient_linear_preserves_population_scaling(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Native expected-sample values use the shared population scaling boundary."""

    def native_result(
        net_benefit: list[list[float]],
        parameter_samples: list[list[float]],
        _trial_sample_size: int,
    ) -> dict[str, object]:
        current = float(np.max(np.mean(net_benefit, axis=0)))
        return {
            "estimator": "efficient_linear",
            "contract_version": 1,
            "expected_current_value": current,
            "expected_sample_value": current + 1.5,
            "expected_perfect_information": current + 2.0,
            "information_fraction": 0.5,
            "evsi": 1.5,
            "sample_count": len(net_benefit),
            "strategy_count": len(net_benefit[0]),
            "parameter_count": len(parameter_samples[0]),
        }

    monkeypatch.setattr(_runtime, "compute_evsi_efficient_linear", native_result)
    result = si_module.evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="efficient",
        metamodel="linear",
        population=10.0,
        discount_rate=0.1,
        time_horizon=2.0,
    )
    annuity = (1.0 - (1.1**-2.0)) / 0.1
    assert result == pytest.approx(1.5 * 10.0 * annuity)


def test_evsi_efficient_random_forest_metamodel(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Test efficient EVSI supports the random forest metamodel setting."""

    evsi_val = evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="efficient",
        metamodel="random_forest",
    )
    assert evsi_val >= 0


def test_evsi_increases_with_sample_size(dummy_psa_for_evsi) -> None:
    """Test that EVSI increases as the trial sample size increases."""

    small_design = TrialDesign(
        arms=[
            DecisionOption(name="New Treatment", sample_size=5),
            DecisionOption(name="Standard Care", sample_size=5),
        ]
    )
    large_design = TrialDesign(
        arms=[
            DecisionOption(name="New Treatment", sample_size=25),
            DecisionOption(name="Standard Care", sample_size=25),
        ]
    )

    small_evsi = evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=small_design,
        method="efficient",
    )
    large_evsi = evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=large_design,
        method="efficient",
    )

    assert small_evsi >= 0
    assert large_evsi >= 0
    assert small_evsi <= large_evsi + 1e-9


def test_evsi_moment_based_method(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Test moment-based EVSI computes a bounded non-negative approximation."""

    evsi_val = evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="moment_based",
    )
    assert evsi_val >= 0


def test_evsi_moment_based_routes_through_native_kernel(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """The public moment estimator uses the versioned Rust result envelope."""
    captured = False

    def compute(
        net_benefit: list[list[float]],
        parameter_samples: list[list[float]],
        trial_sample_size: int,
    ) -> dict[str, object]:
        nonlocal captured
        captured = True
        current = float(np.max(np.mean(net_benefit, axis=0)))
        return {
            "estimator": "moment_based",
            "contract_version": 1,
            "expected_current_value": current,
            "expected_sample_value": current + 1.0,
            "expected_perfect_information": current + 2.0,
            "information_fraction": trial_sample_size
            / (trial_sample_size + len(net_benefit)),
            "evsi": 1.0,
            "sample_count": len(net_benefit),
            "strategy_count": len(net_benefit[0]),
            "parameter_count": len(parameter_samples[0]),
        }

    monkeypatch.setattr(_runtime, "compute_evsi_moment_based", compute)

    result = evsi(
        model_func=deterministic_model_func_evsi,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        method="moment_based",
    )

    assert captured
    assert result == pytest.approx(1.0)


def test_evsi_moment_based_falls_back_for_rank_deficient_design(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Rank-deficient native designs retain the NumPy compatibility path."""

    def rank_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise InputError("moment-based design is rank deficient")

    monkeypatch.setattr(_runtime, "compute_evsi_moment_based", rank_failure)
    with pytest.warns(DeprecationWarning, match="moment-based EVSI fallback"):
        result = evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="moment_based",
        )

    assert result >= 0.0


def test_evsi_moment_based_rejects_malformed_native_envelope(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Malformed native moment output fails closed at the Python boundary."""
    monkeypatch.setattr(
        _runtime,
        "compute_evsi_moment_based",
        lambda *_args, **_kwargs: {"evsi": 1.0},
    )

    with pytest.raises(InputError, match="invalid result envelope"):
        evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="moment_based",
        )


def test_evsi_efficient_rejects_unknown_metamodel(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Test efficient EVSI rejects unsupported metamodel choices."""
    from voiage.exceptions import VoiageNotImplementedError

    with pytest.raises(VoiageNotImplementedError, match="metamodel"):
        evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="efficient",
            metamodel="gaussian_process",
        )


def test_evsi_efficient_requires_sklearn(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Test efficient EVSI reports the scikit-learn dependency clearly."""
    from voiage.exceptions import VoiageNotImplementedError

    monkeypatch.setattr(si_module, "SKLEARN_AVAILABLE", False)
    with pytest.raises(VoiageNotImplementedError, match="scikit-learn"):
        si_module.evsi(
            model_func=deterministic_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="efficient",
            metamodel="random_forest",
        )


def test_evsi_invalid_inputs(dummy_psa_for_evsi, dummy_trial_design_for_evsi) -> None:
    """Test EVSI with various invalid inputs before it hits method implementation."""
    from voiage.exceptions import InputError

    # Invalid model_func
    with pytest.raises(InputError, match="`model_func` must be a callable function"):
        evsi(
            model_func="not_a_function",  # type: ignore
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
        )

    # Invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a ParameterSet object"):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior={"param": np.array([1])},  # type: ignore
            trial_design=dummy_trial_design_for_evsi,
        )

    # Invalid trial_design
    with pytest.raises(InputError, match="`trial_design` must be a TrialDesign object"):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=["not", "a", "trial", "design"],  # type: ignore
        )

    with pytest.raises(InputError, match="n_outer_loops and n_inner_loops"):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            n_outer_loops=0,
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([1.0, 2.0]), "2D net-benefit values"),
        (np.array([[1.0]]), "sample count must match"),
        (np.empty((2, 0)), "at least one strategy"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), "only finite net-benefit values"),
    ],
)
def test_evsi_validates_model_output_contract(values, message) -> None:
    """Stable EVSI diagnostics must reject malformed model output."""
    with pytest.raises(InputError, match=message):
        si_module._validate_net_benefits(values, expected_samples=2)


def test_evsi_parameter_matrix_reports_empty_parameters() -> None:
    """Stable EVSI diagnostics must reject an empty parameter mapping."""
    from types import SimpleNamespace

    with pytest.raises(InputError, match="contain at least one parameter"):
        si_module._parameter_matrix(SimpleNamespace(parameters={}, n_samples=0))


def test_evsi_unknown_method(dummy_psa_for_evsi, dummy_trial_design_for_evsi) -> None:
    """Test EVSI rejects unknown method names after validating inputs."""
    from voiage.exceptions import VoiageNotImplementedError

    with pytest.raises(VoiageNotImplementedError, match="not recognized"):
        evsi(
            model_func=dummy_model_func_evsi,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            method="unknown",
        )


def test_evsi_population_scaling_real_function(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi, monkeypatch
) -> None:
    """Test population scaling and validation in the real EVSI wrapper."""

    def fixed_two_loop(*_args: object, **_kwargs: object) -> float:
        return 200.0

    def fixed_model(psa: PSASample) -> ValueArray:
        nb = np.column_stack(
            [
                np.full(psa.n_samples, 100.0),
                np.full(psa.n_samples, 120.0),
            ]
        )
        return ValueArray.from_numpy(nb, ["current", "new"])

    monkeypatch.setattr(si_module, "_evsi_two_loop", fixed_two_loop)

    result = si_module.evsi(
        model_func=fixed_model,
        psa_prior=dummy_psa_for_evsi,
        trial_design=dummy_trial_design_for_evsi,
        population=1000.0,
        time_horizon=5.0,
        discount_rate=0.0,
    )
    assert result == pytest.approx((200.0 - 120.0) * 1000.0 * 5.0)

    with pytest.raises(InputError, match="Population must be positive"):
        si_module.evsi(
            model_func=fixed_model,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            population=0.0,
            time_horizon=5.0,
        )

    with pytest.raises(InputError, match="Time horizon must be positive"):
        si_module.evsi(
            model_func=fixed_model,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            population=1000.0,
            time_horizon=0.0,
        )

    with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
        si_module.evsi(
            model_func=fixed_model,
            psa_prior=dummy_psa_for_evsi,
            trial_design=dummy_trial_design_for_evsi,
            population=1000.0,
            time_horizon=5.0,
            discount_rate=1.5,
        )


def test_bayesian_update_preserves_unmatched_mean_parameters(
    dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Test posterior update preserves mean parameters without matching trial data."""
    updated = _bayesian_update(
        dummy_psa_for_evsi,
        trial_data={"Different Arm": np.array([1.0, 2.0, 3.0])},
        trial_design=dummy_trial_design_for_evsi,
    )
    assert np.array_equal(
        updated.parameters["mean_new_treatment"],
        dummy_psa_for_evsi.parameters["mean_new_treatment"],
    )


# Mock EVSI for testing population scaling logic within EVSI itself
# This bypasses the NotImplementedError for method logic.
def mock_evsi_for_pop_scaling(
    model_func,
    psa_prior,
    trial_design,  # These are standard EVSI args
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
    method: str = "mocked",  # Method arg still needed
    **kwargs,
) -> float:
    """Mock EVSI to return a fixed per-decision value and implement population scaling."""
    # Validate inputs that population scaling depends on
    if population is not None or time_horizon is not None or discount_rate is not None:
        if not (population is not None and time_horizon is not None):
            raise InputError(
                "To calculate population EVSI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional.",
            )
        if population <= 0:
            raise InputError("Population must be positive.")
        if time_horizon <= 0:
            raise InputError("Time horizon must be positive.")
        if discount_rate is not None and not (0 <= discount_rate <= 1):
            raise InputError("Discount rate must be between 0 and 1.")

    fixed_per_decision_evsi = 10.0  # Arbitrary value for testing scaling

    if (
        population is not None and time_horizon is not None
    ):  # Already validated they are together
        effective_population = population
        if discount_rate is not None:
            if discount_rate == 0:
                annuity_factor = time_horizon
            else:
                annuity_factor = (
                    1 - (1 + discount_rate) ** (-time_horizon)
                ) / discount_rate
            effective_population *= annuity_factor
        else:  # No discount rate
            effective_population *= (
                time_horizon  # Assuming population is annual incidence
            )
        return fixed_per_decision_evsi * effective_population

    return fixed_per_decision_evsi


def test_evsi_population_scaling_logic(
    monkeypatch, dummy_psa_for_evsi, dummy_trial_design_for_evsi
) -> None:
    """Test population scaling logic within EVSI using a mock."""
    # Temporarily replace the real evsi with our mock for this test
    # Note: The sample_information module needs to be imported for monkeypatch to find 'evsi'

    monkeypatch.setattr(si_module, "evsi", mock_evsi_for_pop_scaling)

    fixed_mock_value = 10.0  # Matches the mock's internal per-decision value

    # Test case 1: No population args
    val_no_pop = si_module.evsi(
        dummy_model_func_evsi, dummy_psa_for_evsi, dummy_trial_design_for_evsi
    )
    assert np.isclose(val_no_pop, fixed_mock_value)

    # Test case 2: With population, horizon, no discount
    pop, th = 1000, 5
    val_pop_no_dr = si_module.evsi(
        dummy_model_func_evsi,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        population=pop,
        time_horizon=th,
    )
    assert np.isclose(val_pop_no_dr, fixed_mock_value * pop * th)

    # Test case 3: With population, horizon, and discount rate
    dr = 0.05
    annuity = (1 - (1 + dr) ** (-th)) / dr
    val_pop_dr = si_module.evsi(
        dummy_model_func_evsi,
        dummy_psa_for_evsi,
        dummy_trial_design_for_evsi,
        population=pop,
        time_horizon=th,
        discount_rate=dr,
    )
    assert np.isclose(val_pop_dr, fixed_mock_value * pop * annuity)

    # Test invalid population scaling inputs (should be caught by the mock's validation)
    with pytest.raises(InputError, match="Population must be positive"):
        si_module.evsi(
            dummy_model_func_evsi,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            population=0,
            time_horizon=th,
        )

    with pytest.raises(
        InputError,
        match="To calculate population EVSI, 'population' and 'time_horizon' must be provided",
    ):
        si_module.evsi(
            dummy_model_func_evsi,
            dummy_psa_for_evsi,
            dummy_trial_design_for_evsi,
            population=pop,
            discount_rate=dr,
        )  # Missing time_horizon


# --- Tests for ENBS ---


def test_enbs_zero_cost() -> None:
    """Test ENBS with zero research cost."""
    evsi_val = 500.0
    cost_val = 0.0
    calculated_enbs = enbs(evsi_val, cost_val)
    assert np.isclose(calculated_enbs, evsi_val), (
        "ENBS with zero cost should equal EVSI."
    )


def test_enbs_invalid_inputs() -> None:
    """Test ENBS with various invalid inputs."""
    with pytest.raises(InputError, match="EVSI result must be a number"):
        enbs("not a float", 100.0)  # type: ignore

    with pytest.raises(InputError, match="Research cost must be a number"):
        enbs(1000.0, "not a float")  # type: ignore

    with pytest.raises(InputError, match="Research cost cannot be negative"):
        enbs(1000.0, -50.0)


if __name__ == "__main__":
    pytest.main([__file__])
