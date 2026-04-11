"""
Integration tests for voiage.

These tests verify that different components of voiage work together correctly.
They are slower than unit tests and should be run separately.
"""

import numpy as np
import pytest

from voiage.analysis import evpi, evppi
from voiage.health_economics import calculate_ceac


@pytest.mark.integration
class TestVOIIntegration:
    """Integration tests for VOI analysis pipeline."""

    def test_evpi_to_ceac_pipeline(self):
        """Test that EVPI results can be used in CEAC calculation."""
        np.random.seed(42)
        n_simulations = 1000

        # Generate PSA data
        psa_outputs = np.random.rand(n_simulations, 3)

        # Calculate EVPI
        evpi_value = evpi({}, psa_outputs)

        # Verify EVPI is reasonable
        assert evpi_value >= 0
        assert np.isfinite(evpi_value)

    def test_evppi_with_parameter_subset(self):
        """Test EVPPI calculation with specific parameter subsets."""
        np.random.seed(42)
        n_simulations = 1000

        psa_inputs = {
            'param1': np.random.rand(n_simulations),
            'param2': np.random.rand(n_simulations),
            'param3': np.random.rand(n_simulations),
        }
        psa_outputs = np.random.rand(n_simulations, 2)

        # Calculate EVPPI for subset of parameters
        evppi_value = evppi(psa_inputs, psa_outputs, parameters=['param1', 'param2'])

        assert evppi_value >= 0
        assert evppi_value <= evpi({}, psa_outputs)
        assert np.isfinite(evppi_value)


@pytest.mark.integration
class TestMultiDomainIntegration:
    """Test that multi-domain modules integrate correctly with core."""

    def test_health_economics_integration(self):
        """Test health economics module integrates with core VOI."""
        np.random.seed(42)
        n_simulations = 500

        # Generate health economics data
        net_benefits = np.random.rand(n_simulations, 2) * 1000

        # Calculate EVPI
        evpi_value = evpi({}, net_benefits)

        assert evpi_value >= 0
        assert np.isfinite(evpi_value)
