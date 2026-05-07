# tests/test_network_meta_analysis.py

"""Tests for Network Meta-Analysis VOI methods."""

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods.network_meta_analysis import (
    NetworkMetaAnalysisData,
    calculate_nma_evpi,
    calculate_nma_evppi,
)


@pytest.fixture
def sample_nma_data():
    """Create sample NMA data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_studies = 10
    treatments = ["Placebo", "Drug_A", "Drug_B", "Drug_C"]

    # Create treatment effect samples (vs Placebo as baseline)
    treatment_effects = {
        ("Placebo", "Drug_A"): np.random.normal(0.5, 0.2, n_samples),
        ("Placebo", "Drug_B"): np.random.normal(0.7, 0.25, n_samples),
        ("Placebo", "Drug_C"): np.random.normal(0.3, 0.3, n_samples),
        ("Drug_A", "Drug_B"): np.random.normal(0.2, 0.15, n_samples),
        ("Drug_A", "Drug_C"): np.random.normal(-0.2, 0.18, n_samples),
        ("Drug_B", "Drug_C"): np.random.normal(-0.4, 0.2, n_samples),
    }

    return NetworkMetaAnalysisData(
        treatment_effects=treatment_effects,
        n_studies=n_studies,
        treatments=treatments,
        outcome_type="continuous",
    )


def test_nma_data_creation() -> None:
    """Test basic NetworkMetaAnalysisData creation."""
    np.random.seed(42)
    treatment_effects = {
        ("A", "B"): np.random.rand(100),
        ("A", "C"): np.random.rand(100),
    }

    data = NetworkMetaAnalysisData(
        treatment_effects=treatment_effects,
        n_studies=5,
        treatments=["A", "B", "C"],
        outcome_type="continuous",
    )

    assert data.get_n_treatments() == 3
    assert data.get_n_samples() == 100
    assert data.treatments == ["A", "B", "C"]


def test_nma_data_validation() -> None:
    """Test NetworkMetaAnalysisData input validation."""
    # Empty treatment effects
    with pytest.raises(InputError, match="must not be empty"):
        NetworkMetaAnalysisData(
            treatment_effects={},
            n_studies=1,
            treatments=["A", "B"],
        )

    # Too few treatments
    with pytest.raises(InputError, match="At least 2 treatments"):
        NetworkMetaAnalysisData(
            treatment_effects={("A", "B"): np.array([1, 2, 3])},
            n_studies=1,
            treatments=["A"],
        )

    # Invalid outcome type
    with pytest.raises(InputError, match="must be one of"):
        NetworkMetaAnalysisData(
            treatment_effects={("A", "B"): np.array([1, 2, 3])},
            n_studies=1,
            treatments=["A", "B"],
            outcome_type="invalid",
        )

    with pytest.raises(InputError, match="at least 1"):
        NetworkMetaAnalysisData(
            treatment_effects={("A", "B"): np.array([1, 2, 3])},
            n_studies=0,
            treatments=["A", "B"],
        )

    # Inconsistent sample sizes
    with pytest.raises(InputError, match="same number of samples"):
        NetworkMetaAnalysisData(
            treatment_effects={
                ("A", "B"): np.array([1, 2, 3]),
                ("A", "C"): np.array([1, 2]),
            },
            n_studies=1,
            treatments=["A", "B", "C"],
        )


def test_nma_evpi_basic(sample_nma_data) -> None:
    """Test basic NMA-EVPI calculation."""
    result = calculate_nma_evpi(sample_nma_data, n_samples=500)

    assert isinstance(result, float)
    assert result >= 0  # EVPI should be non-negative


def test_nma_evpi_with_population(sample_nma_data) -> None:
    """Test NMA-EVPI with population scaling."""
    result = calculate_nma_evpi(
        sample_nma_data,
        n_samples=500,
        population=10000,
        time_horizon=10,
        discount_rate=0.03,
    )

    assert isinstance(result, float)
    assert result >= 0


def test_nma_evpi_with_wtp(sample_nma_data) -> None:
    """Test NMA-EVPI with willingness-to-pay threshold."""
    calculate_nma_evpi(sample_nma_data, n_samples=500)
    result_with_wtp = calculate_nma_evpi(
        sample_nma_data,
        n_samples=500,
        willingness_to_pay=50000,
    )

    # With WTP, results should be scaled
    assert result_with_wtp >= 0


def test_nma_evppi_basic(sample_nma_data) -> None:
    """Test basic NMA-EVPPI calculation."""
    # Create parameter samples
    np.random.seed(42)
    parameter_samples = {
        "effect_A": np.random.rand(500),
        "effect_B": np.random.rand(500),
        "effect_C": np.random.rand(500),
    }

    result = calculate_nma_evppi(
        sample_nma_data,
        parameters_of_interest=["effect_A", "effect_B"],
        parameter_samples=parameter_samples,
        n_samples=500,
    )

    assert isinstance(result, float)
    assert result >= 0


def test_nma_evpi_from_dict() -> None:
    """Test NMA-EVPI calculation from dictionary input."""
    np.random.seed(42)
    n_samples = 200

    data_dict = {
        "treatment_effects": {
            "Placebo-Drug_A": np.random.normal(0.5, 0.2, n_samples),
            "Placebo-Drug_B": np.random.normal(0.7, 0.25, n_samples),
        },
        "n_studies": 8,
        "treatments": ["Placebo", "Drug_A", "Drug_B"],
        "outcome_type": "continuous",
    }

    result = calculate_nma_evpi(data_dict, n_samples=200)
    assert isinstance(result, float)
    assert result >= 0


def test_nma_evpi_from_dict_tuple_reverse_and_indirect_effects() -> None:
    """Dictionary conversion should support tuple, reverse, indirect, and no-data paths."""
    effects = np.array([0.2, 0.3, 0.4, 0.5])
    data_dict = {
        "treatment_effects": {
            ("B", "A"): effects,
            ("B", "C"): effects + 0.1,
        },
        "n_studies": 2,
        "treatments": ["A", "B", "C", "D"],
    }

    result = calculate_nma_evpi(data_dict, n_samples=4, willingness_to_pay=10.0)

    assert isinstance(result, float)
    assert result >= 0


def test_nma_dict_rejects_invalid_treatment_effect_key() -> None:
    """Dictionary conversion should reject malformed treatment-effect keys."""
    with pytest.raises(InputError, match="treatment pairs"):
        calculate_nma_evpi(
            {
                "treatment_effects": {"A": np.array([1.0, 2.0])},
                "treatments": ["A", "B"],
            },
            n_samples=2,
        )


def test_nma_evpi_dict_missing_key() -> None:
    """Test NMA-EVPI validation with missing dictionary key."""
    with pytest.raises(InputError, match="must contain 'treatment_effects'"):
        calculate_nma_evpi({"n_studies": 5}, n_samples=100)


def test_nma_evpi_empty() -> None:
    """Test NMA-EVPI with empty data."""
    with pytest.raises(InputError):
        calculate_nma_evpi(
            NetworkMetaAnalysisData(
                treatment_effects={("A", "B"): np.array([])},
                n_studies=1,
                treatments=["A", "B"],
            ),
            n_samples=0,
        )


if __name__ == "__main__":
    pytest.main([__file__])
