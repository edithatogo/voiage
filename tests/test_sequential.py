# tests/test_sequential.py

"""Tests for sequential VOI methods."""

import numpy as np
import pytest

from voiage.exceptions import InputError
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
    if hasattr(psa, 'parameters') and isinstance(psa.parameters, dict):
        evolved_params = {}
        for name, values in psa.parameters.items():
            # Reduce variance to simulate learning
            noise = np.random.normal(0, 0.1, len(values))
            evolved_params[name] = values + noise
        evolved_psa = ParameterSet.from_numpy_or_dict(evolved_params)
        return {"next_psa": evolved_psa}
    return {"next_psa": psa}


def test_sequential_voi_basic():
    """Test basic functionality of sequential_voi."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2, 3])

    # Test with backward induction method
    result = sequential_voi(
        mock_step_model, psa, dyn_spec,
        wtp=50000,
        optimization_method="backward_induction"
    )
    assert isinstance(result, float)
    assert result >= 0  # VOI should be non-negative


def test_sequential_voi_with_population_scaling():
    """Test sequential_voi with population scaling."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    # Test with population scaling
    result = sequential_voi(
        mock_step_model, psa, dyn_spec,
        wtp=50000,
        population=10000,
        time_horizon=10,
        discount_rate=0.03,
        optimization_method="backward_induction"
    )
    assert isinstance(result, float)
    assert result >= 0


def test_sequential_voi_generator():
    """Test sequential_voi with generator method."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    # Test with generator method
    generator = sequential_voi(
        mock_step_model, psa, dyn_spec,
        wtp=50000,
        optimization_method="generator"
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


def test_sequential_voi_edge_cases():
    """Test edge cases for sequential_voi."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })

    # Test with single time step
    dyn_spec_single = DynamicSpec(time_steps=[0])
    result = sequential_voi(
        mock_step_model, psa, dyn_spec_single,
        wtp=50000,
        optimization_method="backward_induction"
    )
    assert isinstance(result, float)
    assert result >= 0

    # Test with invalid dynamic specification type
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model, psa, "not_a_dynamic_spec",
            wtp=50000,
            optimization_method="backward_induction"
        )


def test_sequential_voi_input_validation():
    """Test input validation for sequential_voi."""
    # Create mock PSA samples
    psa = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })

    # Create dynamic specification
    dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    # Test invalid step_model (not callable)
    with pytest.raises(InputError):
        sequential_voi(
            "not_a_function", psa, dyn_spec,
            wtp=50000,
            optimization_method="backward_induction"
        )

    # Test invalid PSA
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model, "not_a_psa", dyn_spec,
            wtp=50000,
            optimization_method="backward_induction"
        )

    # Test invalid wtp
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model, psa, dyn_spec,
            wtp="not_a_number",
            optimization_method="backward_induction"
        )

    # Test invalid population
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model, psa, dyn_spec,
            wtp=50000,
            population=-100,  # Negative population
            optimization_method="backward_induction"
        )

    # Test invalid discount_rate
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model, psa, dyn_spec,
            wtp=50000,
            discount_rate=1.5,  # Rate > 1
            optimization_method="backward_induction"
        )

    # Test invalid optimization_method
    with pytest.raises(InputError):
        sequential_voi(
            mock_step_model, psa, dyn_spec,
            wtp=50000,
            optimization_method="invalid_method"
        )


if __name__ == "__main__":
    pytest.main([__file__])
