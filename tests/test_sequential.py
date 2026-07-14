# tests/test_sequential.py

"""Tests for sequential VOI methods."""

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods import sequential as sequential_module
from voiage.methods.sequential import sequential_voi
from voiage.schema import DynamicSpec, ParameterSet


# Mock step model for testing
def mock_step_model(psa, action, dyn_spec):
    """Return the same PSA with some modifications."""
    # In a real implementation, this would simulate the evolution of uncertainty
    # over time based on the action taken
    return {"next_psa": psa}


def mock_step_model_with_evolution(psa, action, dyn_spec):
    """Step model that simulates evolution of uncertainty."""
    # Add some noise to simulate learning or evolution
    if hasattr(psa, "parameters") and isinstance(psa.parameters, dict):
        evolved_params = {}
        for name, values in psa.parameters.items():
            # Reduce variance to simulate learning
            noise = np.random.normal(0, 0.1, len(values))
            evolved_params[name] = values + noise
        evolved_psa = ParameterSet.from_numpy_or_dict(evolved_params)
        return {"next_psa": evolved_psa}
    return {"next_psa": psa}


def test_sequential_voi_basic() -> None:
    """Test basic functionality of sequential_voi."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2, 3])

    # Test with backward induction method
    result = sequential_voi(
        mock_step_model,
        psa,
        dyn_spec,
        wtp=50000,
        optimization_method="backward_induction",
    )
    assert isinstance(result, float)
    assert result >= 0  # VOI should be non-negative


def test_sequential_voi_with_population_scaling() -> None:
    """Test sequential_voi with population scaling."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    # Test with population scaling
    result = sequential_voi(
        mock_step_model,
        psa,
        dyn_spec,
        wtp=50000,
        population=10000,
        time_horizon=10,
        discount_rate=0.03,
        optimization_method="backward_induction",
    )
    assert isinstance(result, float)
    assert result >= 0


def test_sequential_voi_generator() -> None:
    """Test sequential_voi with generator method."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    # Test with generator method
    generator = sequential_voi(
        mock_step_model, psa, dyn_spec, wtp=50000, optimization_method="generator"
    )

    # Collect results from generator
    results = list(generator)

    assert len(results) == 3  # Should have one entry per time step
    for result in results:
        assert isinstance(result, dict)
        assert "time_step" in result
        assert "current_evpi" in result
        assert "discount_factor" in result
        assert "discounted_evpi" in result
        assert result["current_evpi"] >= 0


def test_sequential_voi_backward_induction_uses_step_model_progression() -> None:
    """Backward-induction mode should advance the PSA state between steps."""
    initial_psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )
    resolved_psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([8.0, 8.0, 8.0, 8.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )

    def learning_step_model(psa, action, dyn_spec):
        _ = psa, action, dyn_spec
        return {"next_psa": resolved_psa}

    result = sequential_voi(
        learning_step_model,
        initial_psa,
        DynamicSpec(time_steps=[0, 1]),
        optimization_method="backward_induction",
    )

    assert result == pytest.approx(2.5)


def test_sequential_evpi_step_matches_standard_formula() -> None:
    """Step EVPI should use E[max NB] - max(E[NB]) for strategy payoffs."""
    psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )

    assert sequential_module._calculate_evpi_at_step(psa, wtp=0.0) == pytest.approx(2.5)


def test_sequential_evpi_step_supports_cost_effect_pairs() -> None:
    """Step EVPI should derive net benefit from strategy cost/effect samples."""
    psa = ParameterSet.from_numpy_or_dict(
        {
            "effect_standard": np.array([1.0, 1.0, 1.0]),
            "cost_standard": np.array([2.0, 2.0, 2.0]),
            "effect_new": np.array([0.0, 3.0, 0.0]),
            "cost_new": np.array([1.0, 1.0, 1.0]),
        }
    )

    # Net benefits at wtp=2 are standard=[0,0,0], new=[-1,5,-1].
    assert sequential_module._calculate_evpi_at_step(psa, wtp=2.0) == pytest.approx(
        2.0 / 3.0
    )


def test_sequential_evpi_step_is_zero_without_payoff_surface() -> None:
    """Plain parameter uncertainty alone should not be treated as EVPI."""
    psa = ParameterSet.from_numpy_or_dict(
        {"param1": np.array([1.0, 3.0, 5.0]), "param2": np.array([2.0, 4.0, 6.0])}
    )

    assert sequential_module._calculate_evpi_at_step(psa, wtp=50000.0) == 0.0


def test_sequential_evpi_step_handles_single_strategy_as_zero() -> None:
    """EVPI should be zero when only one strategy payoff is present."""
    psa = ParameterSet.from_numpy_or_dict(
        {"net_benefit_standard": np.array([1.0, 3.0, 5.0])}
    )

    assert sequential_module._calculate_evpi_at_step(psa, wtp=0.0) == 0.0


def test_sequential_net_benefit_extraction_edge_cases() -> None:
    """Supported extraction helpers should reject malformed payoff surfaces."""
    direct = sequential_module._direct_net_benefits(
        {"net_benefits": np.array([[0.0, 5.0], [10.0, 5.0]])}
    )
    single = sequential_module._direct_net_benefits(
        {"net_benefits": np.array([1.0, 2.0])}
    )
    unsupported = sequential_module._direct_net_benefits(
        {"net_benefits": np.zeros((2, 2, 2))}
    )
    malformed_named = sequential_module._stack_prefixed_strategy_arrays(
        {"net_benefit_a": np.zeros((2, 2))}, prefixes=("net_benefit_",)
    )
    mismatched_cost_effect = sequential_module._cost_effect_net_benefits(
        {
            "effect_a": np.array([1.0, 2.0]),
            "cost_a": np.array([1.0]),
        },
        wtp=1.0,
    )

    assert direct is not None
    np.testing.assert_array_equal(direct, np.array([[0.0, 5.0], [10.0, 5.0]]))
    assert single is not None
    np.testing.assert_array_equal(single, np.array([[1.0], [2.0]]))
    assert unsupported is None
    assert malformed_named is None
    assert mismatched_cost_effect is None


def test_sequential_generator_evpi_decreases_when_learning_resolves_uncertainty() -> (
    None
):
    """Generator EVPI should fall as step updates provide less uncertain payoffs."""
    initial_psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )
    resolved_psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([8.0, 8.0, 8.0, 8.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )

    def learning_step_model(psa, action, dyn_spec):
        return {"next_psa": resolved_psa}

    generator = sequential_voi(
        learning_step_model,
        initial_psa,
        DynamicSpec(time_steps=[0, 1]),
        optimization_method="generator",
    )

    results = list(generator)

    assert results[0]["current_evpi"] == pytest.approx(2.5)
    assert results[1]["current_evpi"] == pytest.approx(0.0)
    assert results[1]["current_evpi"] <= results[0]["current_evpi"]


def test_sequential_generator_discount_and_ignored_invalid_next_state() -> None:
    """Generator should discount current EVPI and ignore invalid next state payloads."""
    psa = ParameterSet.from_numpy_or_dict(
        {
            "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
            "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
        }
    )

    def invalid_next_state_model(psa, action, dyn_spec):
        return {"next_psa": "not a parameter set"}

    generator = sequential_voi(
        invalid_next_state_model,
        psa,
        DynamicSpec(time_steps=[1, 2]),
        discount_rate=0.25,
        optimization_method="generator",
    )

    results = list(generator)

    assert results[0]["discount_factor"] == pytest.approx(0.8)
    assert results[0]["discounted_evpi"] == pytest.approx(2.0)
    assert results[1]["current_evpi"] == pytest.approx(results[0]["current_evpi"])


def test_sequential_voi_edge_cases() -> None:
    """Test edge cases for sequential_voi."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )

    # Test with single time step
    dyn_spec_single = DynamicSpec(time_steps=[0])
    result = sequential_voi(
        mock_step_model,
        psa,
        dyn_spec_single,
        wtp=50000,
        optimization_method="backward_induction",
    )
    assert isinstance(result, float)
    assert result >= 0

    # Test with invalid dynamic specification type
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model,
            psa,
            "not_a_dynamic_spec",
            wtp=50000,
            optimization_method="backward_induction",
        )


def test_sequential_voi_input_validation() -> None:
    """Test input validation for sequential_voi."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    # Test invalid step_model (not callable)
    with pytest.raises(InputError):
        sequential_voi(
            "not_a_function",
            psa,
            dyn_spec,
            wtp=50000,
            optimization_method="backward_induction",
        )

    # Test invalid PSA
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model,
            "not_a_psa",
            dyn_spec,
            wtp=50000,
            optimization_method="backward_induction",
        )

    # Test invalid wtp
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model,
            psa,
            dyn_spec,
            wtp="not_a_number",
            optimization_method="backward_induction",
        )

    # Test invalid population
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model,
            psa,
            dyn_spec,
            wtp=50000,
            population=-100,  # Negative population
            optimization_method="backward_induction",
        )

    # Test invalid discount_rate
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model,
            psa,
            dyn_spec,
            wtp=50000,
            discount_rate=1.5,  # Rate > 1
            optimization_method="backward_induction",
        )

    # Test invalid optimization_method
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model,
            psa,
            dyn_spec,
            wtp=50000,
            optimization_method="invalid_method",
        )


if __name__ == "__main__":
    pytest.main([__file__])
