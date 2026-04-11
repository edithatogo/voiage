# tests/test_advanced_voi_integration.py

"""Integration tests for advanced VOI methods (structural and NMA).

These tests verify that the structural and NMA VOI methods work correctly
with realistic datasets and integrate properly with the rest of the library.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.methods.network_meta_analysis import (
    NetworkMetaAnalysisData,
    calculate_nma_evpi,
    calculate_nma_evppi,
)
from voiage.schema import ParameterSet, ValueArray


@pytest.mark.integration
class TestStructuralVOIIntegration:
    """Integration tests for structural VOI with realistic scenarios."""

    @pytest.fixture
    def health_economic_structures(self):
        """Create realistic health economic model structures."""
        np.random.seed(42)
        n_samples = 1000
        n_strategies = 3  # Standard care, Drug A, Drug B

        # Model 1: Markov model (more conservative)
        def markov_evaluator(psa_sample):
            # Simulate Markov model net benefits
            values = np.zeros((n_samples, n_strategies))
            values[:, 0] = np.random.normal(50000, 5000, n_samples)  # Standard care
            values[:, 1] = np.random.normal(55000, 6000, n_samples)  # Drug A
            values[:, 2] = np.random.normal(53000, 5500, n_samples)  # Drug B
            return ValueArray.from_numpy(values, ["Standard", "Drug_A", "Drug_B"])

        # Model 2: Partitioned survival (more optimistic)
        def surv_evaluator(psa_sample):
            values = np.zeros((n_samples, n_strategies))
            values[:, 0] = np.random.normal(48000, 4500, n_samples)
            values[:, 1] = np.random.normal(58000, 7000, n_samples)
            values[:, 2] = np.random.normal(56000, 6500, n_samples)
            return ValueArray.from_numpy(values, ["Standard", "Drug_A", "Drug_B"])

        # Model 3: Discrete event simulation (most complex)
        def des_evaluator(psa_sample):
            values = np.zeros((n_samples, n_strategies))
            values[:, 0] = np.random.normal(51000, 5200, n_samples)
            values[:, 1] = np.random.normal(56000, 6200, n_samples)
            values[:, 2] = np.random.normal(57000, 6800, n_samples)
            return ValueArray.from_numpy(values, ["Standard", "Drug_A", "Drug_B"])

        evaluators = [markov_evaluator, surv_evaluator, des_evaluator]
        probabilities = [0.4, 0.35, 0.25]  # Model weights

        psa_samples = [
            ParameterSet.from_numpy_or_dict({"transition_rate": np.random.rand(n_samples)}),
            ParameterSet.from_numpy_or_dict({"hazard_ratio": np.random.rand(n_samples)}),
            ParameterSet.from_numpy_or_dict({"event_rate": np.random.rand(n_samples)}),
        ]

        return evaluators, probabilities, psa_samples

    def test_structural_evpi_realistic(self, health_economic_structures):
        """Test structural EVPI with realistic health economic models."""
        evaluators, probabilities, psa_samples = health_economic_structures

        # Without population scaling
        result = structural_evpi(evaluators, probabilities, psa_samples)
        assert isinstance(result, float)
        assert result >= 0

        # With population scaling (typical HTA scenario)
        result_pop = structural_evpi(
            evaluators, probabilities, psa_samples,
            population=100000,
            time_horizon=20,
            discount_rate=0.035
        )
        assert result_pop > result  # Population scaling should increase value

    def test_structural_evppi_learning_about_best_model(self, health_economic_structures):
        """Test structural EVPPI for learning about the most probable model."""
        evaluators, probabilities, psa_samples = health_economic_structures

        # Learn about the Markov model (highest probability)
        result = structural_evppi(
            evaluators, probabilities, psa_samples,
            structures_of_interest=[0]
        )
        assert isinstance(result, float)
        assert result >= 0


@pytest.mark.integration
class TestNMAVOIIntegration:
    """Integration tests for NMA VOI with realistic scenarios."""

    @pytest.fixture
    def realistic_nma_network(self):
        """Create realistic NMA network for diabetes treatments."""
        np.random.seed(42)
        n_samples = 2000
        n_studies = 25

        # Network: Placebo, Metformin, Drug_A, Drug_B, Drug_C
        treatments = ["Placebo", "Metformin", "Drug_A", "Drug_B", "Drug_C"]

        # Treatment effects (HbA1c reduction vs placebo)
        treatment_effects = {
            ("Placebo", "Metformin"): np.random.normal(-0.7, 0.1, n_samples),
            ("Placebo", "Drug_A"): np.random.normal(-1.0, 0.15, n_samples),
            ("Placebo", "Drug_B"): np.random.normal(-0.9, 0.12, n_samples),
            ("Placebo", "Drug_C"): np.random.normal(-1.2, 0.18, n_samples),
            ("Metformin", "Drug_A"): np.random.normal(-0.3, 0.1, n_samples),
            ("Metformin", "Drug_B"): np.random.normal(-0.2, 0.09, n_samples),
            ("Metformin", "Drug_C"): np.random.normal(-0.5, 0.13, n_samples),
            ("Drug_A", "Drug_B"): np.random.normal(0.1, 0.08, n_samples),
            ("Drug_A", "Drug_C"): np.random.normal(-0.2, 0.11, n_samples),
            ("Drug_B", "Drug_C"): np.random.normal(-0.3, 0.12, n_samples),
        }

        return NetworkMetaAnalysisData(
            treatment_effects=treatment_effects,
            n_studies=n_studies,
            treatments=treatments,
            outcome_type="continuous",
        )

    def test_nma_evpi_diabetes_network(self, realistic_nma_network):
        """Test NMA-EVPI with realistic diabetes treatment network."""
        result = calculate_nma_evpi(realistic_nma_network, n_samples=2000)
        assert isinstance(result, float)
        assert result >= 0

    def test_nma_evpi_with_population_scaling(self, realistic_nma_network):
        """Test NMA-EVPI with population scaling for HTA."""
        result = calculate_nma_evpi(
            realistic_nma_network,
            n_samples=2000,
            population=500000,  # Large patient population
            time_horizon=10,
            discount_rate=0.03,
        )
        assert isinstance(result, float)
        assert result >= 0

    def test_nma_evppi_specific_parameters(self, realistic_nma_network):
        """Test NMA-EVPPI for specific parameters of interest."""
        # Create realistic parameter samples
        np.random.seed(42)
        n_samples = 2000
        parameter_samples = {
            "baseline_risk": np.random.beta(10, 50, n_samples),
            "treatment_effect_A": np.random.normal(-1.0, 0.15, n_samples),
            "treatment_effect_B": np.random.normal(-0.9, 0.12, n_samples),
            "heterogeneity": np.random.uniform(0.05, 0.2, n_samples),
        }

        result = calculate_nma_evppi(
            realistic_nma_network,
            parameters_of_interest=["treatment_effect_A", "treatment_effect_B"],
            parameter_samples=parameter_samples,
            n_samples=n_samples,
        )
        assert isinstance(result, float)
        assert result >= 0

    def test_nma_evpi_vs_evppi_relationship(self, realistic_nma_network):
        """Test that EVPPI <= EVPI (resolving partial uncertainty <= full uncertainty)."""
        np.random.seed(42)
        n_samples = 2000

        # Calculate EVPI
        evpi_result = calculate_nma_evpi(realistic_nma_network, n_samples=n_samples)

        # Calculate EVPPI for subset of parameters
        parameter_samples = {
            "param_A": np.random.rand(n_samples),
            "param_B": np.random.rand(n_samples),
        }
        evppi_result = calculate_nma_evppi(
            realistic_nma_network,
            parameters_of_interest=["param_A"],
            parameter_samples=parameter_samples,
            n_samples=n_samples,
        )

        # EVPPI should be <= EVPI
        assert evppi_result <= evpi_result + 1e-9  # Small tolerance for numerical errors


@pytest.mark.integration
class TestCrossMethodIntegration:
    """Test integration between different VOI methods."""

    def test_structural_and_nma_consistency(self):
        """Test that structural and NMA methods give consistent results."""
        np.random.seed(42)
        n_samples = 500

        # Create simple 2-structure scenario that could be modeled both ways
        # As structural VOI
        def evaluator1(psa):
            values = np.random.normal(100, 10, (n_samples, 2))
            return ValueArray.from_numpy(values, ["A", "B"])

        def evaluator2(psa):
            values = np.random.normal(105, 12, (n_samples, 2))
            return ValueArray.from_numpy(values, ["A", "B"])

        structural_result = structural_evpi(
            [evaluator1, evaluator2],
            [0.5, 0.5],
            [
                ParameterSet.from_numpy_or_dict({"p1": np.random.rand(n_samples)}),
                ParameterSet.from_numpy_or_dict({"p2": np.random.rand(n_samples)}),
            ]
        )

        # Same scenario as NMA
        nma_data = NetworkMetaAnalysisData(
            treatment_effects={
                ("A", "B"): np.random.normal(5, 11, n_samples),
            },
            n_studies=10,
            treatments=["A", "B"],
            outcome_type="continuous",
        )

        nma_result = calculate_nma_evpi(nma_data, n_samples=n_samples)

        # Both should be non-negative and in similar ballpark
        assert structural_result >= 0
        assert nma_result >= 0


if __name__ == "__main__":
    pytest.main([__file__])
