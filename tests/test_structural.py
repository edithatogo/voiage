# tests/test_structural.py

"""Tests for structural VOI methods."""

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.schema import ParameterSet, ValueArray


# Mock model structure evaluators for testing
def mock_evaluator1(psa_sample):
    """Return fixed net benefits."""
    # 100 samples, 2 strategies
    values = np.random.rand(100, 2) * 100
    # Make strategy 0 better on average
    values[:, 0] += 50
    return ValueArray.from_numpy(values, ["Strategy A", "Strategy B"])


def mock_evaluator2(psa_sample):
    """Another simple evaluator."""
    # 100 samples, 2 strategies
    values = np.random.rand(100, 2) * 100
    # Make strategy 1 better on average
    values[:, 1] += 50
    return ValueArray.from_numpy(values, ["Strategy A", "Strategy B"])


def mock_evaluator3(psa_sample):
    """Third evaluator with different characteristics."""
    # 100 samples, 2 strategies
    values = np.random.rand(100, 2) * 100
    # Make both strategies similar
    return ValueArray.from_numpy(values, ["Strategy A", "Strategy B"])


@pytest.fixture
def sample_structures():
    """Create sample model structures for testing."""
    evaluators = [mock_evaluator1, mock_evaluator2, mock_evaluator3]
    probabilities = [0.5, 0.3, 0.2]

    # Create mock PSA samples
    psa1 = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })
    psa2 = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })
    psa3 = ParameterSet.from_numpy_or_dict({
        "param1": np.random.rand(100),
        "param2": np.random.rand(100)
    })
    psa_samples = [psa1, psa2, psa3]

    return evaluators, probabilities, psa_samples


def test_structural_evpi_basic(sample_structures):
    """Test basic functionality of structural_evpi."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test without population scaling
    result = structural_evpi(evaluators, probabilities, psa_samples)
    assert isinstance(result, float)
    assert result >= 0  # EVPI should be non-negative


def test_structural_evpi_with_population_scaling(sample_structures):
    """Test structural_evpi with population scaling."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test with population scaling
    result = structural_evpi(
        evaluators, probabilities, psa_samples,
        population=10000,
        time_horizon=10,
        discount_rate=0.03
    )
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evpi_edge_cases():
    """Test edge cases for structural_evpi."""
    # Test with empty lists
    result = structural_evpi([], [], [])
    assert result == 0.0

    # Test with single structure (no uncertainty, but probabilities sum to 1)
    result = structural_evpi([mock_evaluator1], [1.0], [
        ParameterSet.from_numpy_or_dict({
            "param1": np.random.rand(100),
            "param2": np.random.rand(100)
        })
    ])
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evpi_input_validation(sample_structures):
    """Test input validation for structural_evpi."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test mismatched list lengths
    with pytest.raises(InputError):
        structural_evpi(evaluators[:-1], probabilities, psa_samples)

    with pytest.raises(InputError):
        structural_evpi(evaluators, probabilities[:-1], psa_samples)

    with pytest.raises(InputError):
        structural_evpi(evaluators, probabilities, psa_samples[:-1])

    # Test probabilities not summing to 1
    with pytest.raises(InputError):
        structural_evpi([mock_evaluator1, mock_evaluator2], [0.5, 0.3],
                       [ParameterSet.from_numpy_or_dict({"p": np.array([1])}),
                        ParameterSet.from_numpy_or_dict({"p": np.array([2])})])


def test_structural_evppi_basic(sample_structures):
    """Test basic functionality of structural_evppi."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test learning about first structure
    result = structural_evppi(evaluators, probabilities, psa_samples, [0])
    assert isinstance(result, float)
    assert result >= 0

    # Test learning about multiple structures
    result = structural_evppi(evaluators, probabilities, psa_samples, [0, 1])
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evppi_edge_cases():
    """Test edge cases for structural_evppi."""
    # Test with empty structures_of_interest (should return 0)
    result = structural_evppi([mock_evaluator1], [1.0],
                            [ParameterSet.from_numpy_or_dict({
                                "param1": np.random.rand(100),
                                "param2": np.random.rand(100)
                            })], [])
    assert result == 0.0


def test_structural_evppi_input_validation(sample_structures):
    """Test input validation for structural_evppi."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test invalid structure indices
    with pytest.raises(InputError):
        structural_evppi(evaluators, probabilities, psa_samples, [10])  # Index out of range

    with pytest.raises(InputError):
        structural_evppi(evaluators, probabilities, psa_samples, [-1])  # Negative index


if __name__ == "__main__":
    pytest.main([__file__])
