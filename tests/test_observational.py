# tests/test_observational.py

"""Tests for observational study VOI methods."""

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.methods import observational as obs_module
from voiage.methods.observational import (
    basic_observational_study_modeler,
    voi_observational,
)
from voiage.schema import ParameterSet, ValueArray


# Mock observational study modeler for testing
def mock_obs_modeler(psa_samples, study_design, bias_models):
    """Mock observational study modeler that returns simple net benefits."""
    # Create simple net benefits based on PSA samples
    n_samples = psa_samples.n_samples
    # Create net benefits for 2 strategies
    nb_values = np.random.rand(n_samples, 2) * 1000
    # Make strategy 1 slightly better on average
    nb_values[:, 1] += 100

    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(2),
            "strategy": ("n_strategies", ["Standard Care", "New Treatment"]),
        },
    )
    return ValueArray(dataset=dataset)


@pytest.fixture
def sample_psa():
    """Create a sample ParameterSet for testing."""
    params = {
        "effectiveness": np.random.normal(0.7, 0.1, 50),
        "cost": np.random.normal(5000, 500, 50),
    }
    return ParameterSet.from_numpy_or_dict(params)


@pytest.fixture
def sample_study_design():
    """Create a sample observational study design for testing."""
    return {
        "study_type": "cohort",
        "sample_size": 1000,
        "variables_collected": ["treatment", "outcome"],
    }


@pytest.fixture
def sample_bias_models():
    """Create sample bias models for testing."""
    return {"confounding": {"strength": 0.2}, "selection_bias": {"probability": 0.1}}


def test_voi_observational_basic(
    sample_psa, sample_study_design, sample_bias_models
) -> None:
    """Test basic functionality of voi_observational."""
    result = voi_observational(
        obs_study_modeler=mock_obs_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        n_outer_loops=5,
    )

    assert isinstance(result, float)
    assert result >= 0  # VOI should be non-negative


def test_voi_observational_with_population_scaling(
    sample_psa, sample_study_design, sample_bias_models
) -> None:
    """Test voi_observational with population scaling."""
    result = voi_observational(
        obs_study_modeler=mock_obs_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5,
    )

    assert isinstance(result, float)
    assert result >= 0


def test_voi_observational_input_validation(
    sample_psa, sample_study_design, sample_bias_models
) -> None:
    """Test input validation for voi_observational."""
    # Test invalid obs_study_modeler
    with pytest.raises(
        InputError, match="`obs_study_modeler` must be a callable function"
    ):
        voi_observational(
            obs_study_modeler="not_a_function",
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
        )

    # Test invalid psa_prior
    with pytest.raises(InputError, match="`psa_prior` must be a PSASample object"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior="not_a_psa",
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
        )

    # Test invalid observational_study_design
    with pytest.raises(
        InputError, match="`observational_study_design` must be a dictionary"
    ):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design="not_a_dict",
            bias_models=sample_bias_models,
        )

    # Test invalid bias_models
    with pytest.raises(InputError, match="`bias_models` must be a dictionary"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models="not_a_dict",
        )

    # Test invalid loop parameters
    with pytest.raises(InputError, match="n_outer_loops must be positive"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            n_outer_loops=0,
        )


def test_voi_observational_population_scaling_validation(
    sample_psa, sample_study_design, sample_bias_models
) -> None:
    """Test population scaling validation in voi_observational."""
    # Test invalid population
    with pytest.raises(InputError, match="Population must be positive"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            population=0,
            time_horizon=10,
        )

    # Test invalid time_horizon
    with pytest.raises(InputError, match="Time horizon must be positive"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            population=1000,
            time_horizon=0,
        )

    # Test invalid discount_rate
    with pytest.raises(InputError, match="Discount rate must be between 0 and 1"):
        voi_observational(
            obs_study_modeler=mock_obs_modeler,
            psa_prior=sample_psa,
            observational_study_design=sample_study_design,
            bias_models=sample_bias_models,
            population=1000,
            time_horizon=10,
            discount_rate=1.5,
        )


def test_voi_observational_edge_cases(
    sample_psa, sample_study_design, sample_bias_models
) -> None:
    """Test edge cases for voi_observational."""
    # Test with very small number of loops
    result = voi_observational(
        obs_study_modeler=mock_obs_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        n_outer_loops=1,
    )

    assert isinstance(result, float)
    assert result >= 0


def test_basic_observational_modeler_returns_net_benefit_array() -> None:
    """Built-in modeler should work with explicit strategy net benefits."""
    psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )

    prior = basic_observational_study_modeler(
        psa,
        {"sample_size": 100},
        {},
    )
    posterior = basic_observational_study_modeler(
        psa,
        {
            "sample_size": 100,
            "_true_parameters": {
                "net_benefit_standard": 10.0,
                "net_benefit_new": 5.0,
            },
        },
        {"confounding": {"strength": 0.1}},
    )

    assert prior.numpy_values.shape == (4, 2)
    assert posterior.numpy_values.shape == (4, 2)
    assert np.var(posterior.numpy_values[:, 1]) < np.var(prior.numpy_values[:, 1])


def test_voi_observational_uses_default_modeler_for_net_benefit_psa() -> None:
    """voi_observational should run without a user-supplied modeler."""
    psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )

    result = voi_observational(
        psa_prior=psa,
        observational_study_design={"study_type": "cohort", "sample_size": 100},
        bias_models={},
        n_outer_loops=4,
    )

    assert result >= 0.0


def test_basic_observational_modeler_supports_cost_effect_pairs() -> None:
    """Built-in modeler should derive strategy net benefits from cost/effect pairs."""
    psa = ParameterSet.from_numpy_or_dict(
        {
            "effect_standard": np.array([1.0, 1.0, 1.0]),
            "cost_standard": np.array([2.0, 2.0, 2.0]),
            "effect_new": np.array([0.0, 3.0, 0.0]),
            "cost_new": np.array([1.0, 1.0, 1.0]),
        }
    )

    result = basic_observational_study_modeler(
        psa,
        {"sample_size": 25, "wtp": 2.0},
        {"selection_bias": 0.2},
    )

    np.testing.assert_array_equal(
        result.numpy_values,
        np.array([[-1.0, 0.0], [5.0, 0.0], [-1.0, 0.0]]),
    )


def test_basic_observational_modeler_validates_payoff_inputs(
    sample_psa,
) -> None:
    """Built-in modeler should reject missing or malformed payoff surfaces."""
    with pytest.raises(InputError, match="requires explicit strategy net benefits"):
        basic_observational_study_modeler(sample_psa, {"sample_size": 10}, {})

    with pytest.raises(InputError, match="one-dimensional"):
        obs_module._stack_named_strategy_arrays(
            {"net_benefit_a": np.zeros((2, 2))},
            prefixes=("net_benefit_",),
        )

    with pytest.raises(InputError, match="matching 1D arrays"):
        obs_module._cost_effect_strategy_net_benefits(
            {"effect_a": np.array([1.0, 2.0]), "cost_a": np.zeros((2, 1))},
            wtp=1.0,
        )


def test_voi_observational_uses_prior_value_when_post_modeler_fails(
    sample_psa,
    sample_study_design,
    sample_bias_models,
) -> None:
    """Post-study modeler failures should fall back to the current information value."""
    calls = 0

    def partly_failing_modeler(psa_samples, study_design, bias_models):
        nonlocal calls
        calls += 1
        if calls > 1:
            msg = "post-study model failed"
            raise RuntimeError(msg)
        return mock_obs_modeler(psa_samples, study_design, bias_models)

    result = voi_observational(
        obs_study_modeler=partly_failing_modeler,
        psa_prior=sample_psa,
        observational_study_design=sample_study_design,
        bias_models=sample_bias_models,
        n_outer_loops=2,
    )

    assert result == pytest.approx(0.0)


def test_observational_helper_edge_branches() -> None:
    """Small helper branches should keep deterministic behavior."""
    assert obs_module._combined_bias_strength(
        {
            "confounding": {"strength": -0.2, "probability": 0.1},
            "selection": 0.3,
            "ignored": object(),
        }
    ) == pytest.approx(0.6)
    assert obs_module._true_strategy_net_benefit(
        "New Treatment", {"nb_new_treatment": 12.0}
    ) == pytest.approx(12.0)
    assert obs_module._true_strategy_net_benefit("Missing", {}) == 0.0
    assert obs_module._remove_first_prefix("plain", ("net_benefit_",)) == "plain"


if __name__ == "__main__":
    pytest.main([__file__])
