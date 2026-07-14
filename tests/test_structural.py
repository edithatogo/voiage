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


@pytest.fixture()
def sample_structures():
    """Create sample model structures for testing."""
    evaluators = [mock_evaluator1, mock_evaluator2, mock_evaluator3]
    probabilities = [0.5, 0.3, 0.2]

    # Create mock PSA samples
    psa1 = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )
    psa2 = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )
    psa3 = ParameterSet.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )
    psa_samples = [psa1, psa2, psa3]

    return evaluators, probabilities, psa_samples


def test_structural_evpi_basic(sample_structures) -> None:
    """Test basic functionality of structural_evpi."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test without population scaling
    result = structural_evpi(evaluators, probabilities, psa_samples)
    assert isinstance(result, float)
    assert result >= 0  # EVPI should be non-negative


def test_structural_evpi_with_population_scaling(sample_structures) -> None:
    """Test structural_evpi with population scaling."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test with population scaling
    result = structural_evpi(
        evaluators,
        probabilities,
        psa_samples,
        population=10000,
        time_horizon=10,
        discount_rate=0.03,
    )
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evpi_edge_cases() -> None:
    """Test edge cases for structural_evpi."""
    # Test with empty lists
    result = structural_evpi([], [], [])
    assert result == 0.0

    # Test with single structure (no uncertainty, but probabilities sum to 1)
    result = structural_evpi(
        [mock_evaluator1],
        [1.0],
        [
            ParameterSet.from_numpy_or_dict(
                {"param1": np.random.rand(100), "param2": np.random.rand(100)}
            )
        ],
    )
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evpi_input_validation(sample_structures) -> None:
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
        structural_evpi(
            [mock_evaluator1, mock_evaluator2],
            [0.5, 0.3],
            [
                ParameterSet.from_numpy_or_dict({"p": np.array([1])}),
                ParameterSet.from_numpy_or_dict({"p": np.array([2])}),
            ],
        )

    with pytest.raises(InputError, match="Population"):
        structural_evpi(
            evaluators, probabilities, psa_samples, population=0, time_horizon=5
        )

    with pytest.raises(InputError, match="Time horizon"):
        structural_evpi(
            evaluators, probabilities, psa_samples, population=100, time_horizon=0
        )

    with pytest.raises(InputError, match="Discount rate"):
        structural_evpi(
            evaluators,
            probabilities,
            psa_samples,
            population=100,
            time_horizon=5,
            discount_rate=2.0,
        )

    with pytest.raises(InputError, match="population.*time_horizon"):
        structural_evpi(evaluators, probabilities, psa_samples, population=100)


def test_structural_evppi_basic(sample_structures) -> None:
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


def test_structural_evppi_edge_cases() -> None:
    """Test edge cases for structural_evppi."""
    # Test with empty structures_of_interest (should return 0)
    result = structural_evppi(
        [mock_evaluator1],
        [1.0],
        [
            ParameterSet.from_numpy_or_dict(
                {"param1": np.random.rand(100), "param2": np.random.rand(100)}
            )
        ],
        [],
    )
    assert result == 0.0


def test_structural_evppi_input_validation(sample_structures) -> None:
    """Test input validation for structural_evppi."""
    evaluators, probabilities, psa_samples = sample_structures

    # Test invalid structure indices
    with pytest.raises(InputError):
        structural_evppi(
            evaluators, probabilities, psa_samples, [10]
        )  # Index out of range

    with pytest.raises(InputError):
        structural_evppi(evaluators, probabilities, psa_samples, [-1])  # Negative index

    with pytest.raises(InputError, match="Structure probabilities"):
        structural_evppi(evaluators[:2], [0.5, 0.3], psa_samples[:2], [0])

    with pytest.raises(InputError, match="Population"):
        structural_evppi(
            evaluators, probabilities, psa_samples, [0], population=0, time_horizon=5
        )

    with pytest.raises(InputError, match="Time horizon"):
        structural_evppi(
            evaluators, probabilities, psa_samples, [0], population=100, time_horizon=0
        )

    with pytest.raises(InputError, match="Discount rate"):
        structural_evppi(
            evaluators,
            probabilities,
            psa_samples,
            [0],
            population=100,
            time_horizon=5,
            discount_rate=2.0,
        )

    with pytest.raises(InputError, match="population.*time_horizon"):
        structural_evppi(evaluators, probabilities, psa_samples, [0], population=100)


# --- Tests for JIT-compiled versions ---


def test_structural_evpi_jit_basic() -> None:
    """Test basic functionality of JIT-compiled structural_evpi."""
    try:
        from voiage.methods.structural import structural_evpi_jit
    except InputError:
        pytest.skip("JAX not available")

    np.random.seed(42)
    n_samples = 100
    n_strategies = 2
    n_structures = 3

    # Create sample net benefit arrays
    nb_arrays = [
        np.random.rand(n_samples, n_strategies) * 100 for _ in range(n_structures)
    ]
    probabilities = [0.5, 0.3, 0.2]

    result = structural_evpi_jit(nb_arrays, probabilities)
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evpi_jit_with_population() -> None:
    """Test JIT-compiled structural_evpi with population scaling."""
    try:
        from voiage.methods.structural import structural_evpi_jit
    except InputError:
        pytest.skip("JAX not available")

    np.random.seed(42)
    nb_arrays = [np.random.rand(100, 2) * 100 for _ in range(2)]
    probabilities = [0.6, 0.4]

    result = structural_evpi_jit(
        nb_arrays, probabilities, population=10000, time_horizon=10, discount_rate=0.03
    )
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evppi_jit_basic() -> None:
    """Test basic functionality of JIT-compiled structural_evppi."""
    try:
        from voiage.methods.structural import structural_evppi_jit
    except InputError:
        pytest.skip("JAX not available")

    np.random.seed(42)
    nb_arrays = [np.random.rand(100, 2) * 100 for _ in range(3)]
    probabilities = [0.5, 0.3, 0.2]

    result = structural_evppi_jit(nb_arrays, probabilities, [0])
    assert isinstance(result, float)
    assert result >= 0


def test_structural_evppi_jit_all_structures() -> None:
    """Test JIT-compiled structural_evppi when all structures are of interest (should equal SEVPI)."""
    try:
        from voiage.methods.structural import structural_evpi_jit, structural_evppi_jit
    except InputError:
        pytest.skip("JAX not available")

    np.random.seed(42)
    nb_arrays = [np.random.rand(100, 2) * 100 for _ in range(2)]
    probabilities = [0.6, 0.4]

    evpi_result = structural_evpi_jit(nb_arrays, probabilities)
    evppi_result = structural_evppi_jit(nb_arrays, probabilities, [0, 1])

    # When all structures are of interest, EVPPI should equal EVPI
    np.testing.assert_allclose(evpi_result, evppi_result, rtol=1e-5)


def test_structural_jit_edge_validation() -> None:
    """Cover JIT structural validation and empty-input branches."""
    from voiage.methods.structural import structural_evpi_jit, structural_evppi_jit

    assert structural_evpi_jit([], []) == 0.0
    assert structural_evppi_jit([np.array([[1.0, 2.0]])], [1.0], []) == 0.0

    with pytest.raises(InputError, match="must match"):
        structural_evpi_jit([np.array([[1.0, 2.0]])], [0.5, 0.5])

    with pytest.raises(InputError, match="sum to 1"):
        structural_evpi_jit([np.array([[1.0, 2.0]])], [0.5])

    with pytest.raises(InputError, match="must match"):
        structural_evppi_jit([np.array([[1.0, 2.0]])], [0.5, 0.5], [0])

    with pytest.raises(InputError, match="sum to 1"):
        structural_evppi_jit([np.array([[1.0, 2.0]])], [0.5], [0])


if __name__ == "__main__":
    pytest.main([__file__])
