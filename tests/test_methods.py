"""Unit tests for the core VOI methods (EVPI, EVPPI, EVSI, ENBS)."""

import numpy as np
import pytest

from pyvoi.core.data_structures import NetBenefitArray, PSASample, TrialDesign
from pyvoi.exceptions import DimensionMismatchError, InputError
from pyvoi.methods.basic import evpi, evppi
from pyvoi.methods.sample_information import enbs, evsi

# --- Tests for EVPI ---


def test_evpi_basic_calculation(sample_net_benefit_array_2strat: NetBenefitArray):
    """Test EVPI with a simple, deterministic scenario."""
    # Expected calculation: E[max(NB)] - max(E[NB])
    # NB values: [[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]]
    # Max per sample: [105, 110, 110, 120, 115]
    # E[max(NB)] = (105 + 110 + 110 + 120 + 115) / 5 = 560 / 5 = 112
    # E[NB_stratA] = (100 + 110 + 90 + 120 + 95) / 5 = 515 / 5 = 103
    # E[NB_stratB] = (105 + 100 + 110 + 100 + 115) / 5 = 530 / 5 = 106
    # max(E[NB]) = max(103, 106) = 106
    # EVPI = 112 - 106 = 6
    expected_evpi = 6.0
    calculated_evpi = evpi(sample_net_benefit_array_2strat)
    np.testing.assert_almost_equal(calculated_evpi, expected_evpi, decimal=5)


def test_evpi_no_uncertainty():
    """Test EVPI when there is no uncertainty (all samples are identical)."""
    nb_values = np.array([[10, 12], [10, 12], [10, 12]])
    nba = NetBenefitArray(values=nb_values)
    expected_evpi = 0.0
    calculated_evpi = evpi(nba)
    np.testing.assert_almost_equal(calculated_evpi, expected_evpi, decimal=5)


def test_evpi_single_strategy():
    """Test EVPI with a single strategy (should be 0)."""
    nb_values = np.array([[10], [15], [8]])
    nba = NetBenefitArray(values=nb_values)
    expected_evpi = 0.0
    calculated_evpi = evpi(nba)
    np.testing.assert_almost_equal(calculated_evpi, expected_evpi, decimal=5)


def test_evpi_input_validation():
    """Test EVPI input validation for incorrect types and dimensions."""
    with pytest.raises(
        InputError, match="Input must be an instance of NetBenefitArray."
    ):
        evpi(None)  # type: ignore
    with pytest.raises(
        InputError, match="Input must be an instance of NetBenefitArray."
    ):
        evpi(np.array([[1, 2]]))  # type: ignore

    # Test with NetBenefitArray having wrong dimensions (e.g., 1D or 3D)
    nba_1d = NetBenefitArray(
        values=np.array([1, 2, 3]).reshape(-1, 1)
    )  # Reshape to make it 2D (n_samples, 1_strategy)
    # This case should actually pass as it's (samples, strategies) where strategies = 1
    evpi(nba_1d)  # Should not raise error

    # Create a NetBenefitArray that is explicitly 3D (which should be caught by evpi, not NBA constructor)
    class MockNetBenefitArray(NetBenefitArray):
        def __post_init__(self: "MockNetBenefitArray") -> None:
            # Bypass original __post_init__ to force 3D for testing evpi's check
            object.__setattr__(self, "values", self.values)

    mock_nba_3d = MockNetBenefitArray(values=np.random.rand(5, 2, 3))
    with pytest.raises(
        DimensionMismatchError, match="NetBenefitArray values must be a 2D array"
    ):
        evpi(mock_nba_3d)


# --- Tests for EVPPI (Placeholder) ---


def test_evppi_placeholder(
    sample_net_benefit_array_2strat: NetBenefitArray, sample_psa_2params: PSASample
):
    """Test EVPPI placeholder returns 0.0 and raises NotImplementedError if not implemented."""
    # This test will need to be updated once EVPPI is fully implemented.
    # For now, it checks the placeholder behavior.
    param_of_interest = "param1"
    calculated_evppi = evppi(
        sample_net_benefit_array_2strat, sample_psa_2params, param_of_interest
    )
    np.testing.assert_almost_equal(
        calculated_evppi, 0.0, decimal=5
    )  # Placeholder return value

    # If we decide to raise NotImplementedError in the placeholder, uncomment this:
    # with pytest.raises(NotImplementedError, match="EVPPI calculation is a placeholder."):
    #     evppi(sample_net_benefit_array_2strat, sample_psa_2params, param_of_interest)


def test_evppi_input_validation(
    sample_net_benefit_array_2strat: NetBenefitArray, sample_psa_2params: PSASample
):
    """Test EVPPI input validation for incorrect types and missing parameters."""
    with pytest.raises(
        InputError, match="Input must be an instance of NetBenefitArray."
    ):
        evppi(None, sample_psa_2params, "param1")  # type: ignore
    with pytest.raises(
        InputError, match="'psa_sample' must be an instance of PSASample."
    ):
        evppi(sample_net_benefit_array_2strat, None, "param1")  # type: ignore
    with pytest.raises(
        InputError,
        match="'parameters_of_interest' must be a string or a list of strings.",
    ):
        evppi(sample_net_benefit_array_2strat, sample_psa_2params, 123)  # type: ignore
    with pytest.raises(
        InputError, match=r"Parameter\(s\) 'non_existent_param' not found in PSASample."
    ):
        evppi(sample_net_benefit_array_2strat, sample_psa_2params, "non_existent_param")
    with pytest.raises(
        InputError,
        match=r"Parameter\(s\) 'param1, non_existent_param' not found in PSASample.",
    ):
        evppi(
            sample_net_benefit_array_2strat,
            sample_psa_2params,
            ["param1", "non_existent_param"],
        )


# --- Tests for EVSI (Placeholder) ---


def test_evsi_placeholder(
    sample_net_benefit_array_2strat: NetBenefitArray,
    sample_psa_2params: PSASample,
    sample_trial_design_2arms: TrialDesign,
):
    """Test EVSI placeholder returns 0.0 and raises NotImplementedError if not implemented."""

    def dummy_model(psa: PSASample) -> NetBenefitArray:
        return NetBenefitArray(
            values=np.array([[10, 12], [15, 11], [8, 10], [13, 14], [9, 7]])
        )

    calculated_evsi = evsi(
        sample_net_benefit_array_2strat,
        sample_psa_2params,
        sample_trial_design_2arms,
        dummy_model,
    )
    np.testing.assert_almost_equal(
        calculated_evsi, 0.0, decimal=5
    )  # Placeholder return value

    # If we decide to raise NotImplementedError in the placeholder, uncomment this:
    # with pytest.raises(NotImplementedError, match="EVSI calculation is a placeholder."):
    #     evsi(sample_net_benefit_array_2strat, sample_psa_2params, sample_trial_design_2arms, dummy_model)


def test_evsi_input_validation(
    sample_net_benefit_array_2strat: NetBenefitArray,
    sample_psa_2params: PSASample,
    sample_trial_design_2arms: TrialDesign,
):
    """Test EVSI input validation for incorrect types."""
    pass

    def dummy_model(psa: PSASample) -> NetBenefitArray:
        return NetBenefitArray(values=np.array([[10, 12]]))

    with pytest.raises(
        InputError, match="'net_benefit_array' must be an instance of NetBenefitArray."
    ):
        evsi(None, sample_psa_2params, sample_trial_design_2arms, dummy_model)  # type: ignore
    with pytest.raises(
        InputError, match="'psa_sample' must be an instance of PSASample."
    ):
        evsi(
            sample_net_benefit_array_2strat,
            dummy_model,
            sample_trial_design_2arms,
        )  # type: ignore
    with pytest.raises(
        InputError, match="'trial_design' must be an instance of TrialDesign."
    ):
        evsi(sample_net_benefit_array_2strat, sample_psa_2params, None, dummy_model)  # type: ignore
    with pytest.raises(
        InputError, match="'model_function' must be a callable function."
    ):
        evsi(
            sample_net_benefit_array_2strat,
            sample_psa_2params,
            sample_trial_design_2arms,
            "not_a_function",
        )  # type: ignore


# --- Tests for ENBS (Placeholder) ---


def test_enbs_placeholder(
    sample_net_benefit_array_2strat: NetBenefitArray,
    sample_psa_2params: PSASample,
    sample_trial_design_2arms: TrialDesign,
):
    """Test ENBS placeholder returns EVSI - cost_of_study."""
    pass

    def dummy_model(psa: PSASample) -> NetBenefitArray:
        return NetBenefitArray(
            values=np.array([[10, 12], [15, 11], [8, 10], [13, 14], [9, 7]])
        )

    cost = 1000.0
    calculated_enbs = enbs(
        sample_net_benefit_array_2strat,
        sample_trial_design_2arms,
        cost,
        dummy_model,
        sample_psa_2params,
    )
    np.testing.assert_almost_equal(
        calculated_enbs, -cost, decimal=5
    )  # Placeholder return value (0.0 - cost)

    # If we decide to raise NotImplementedError in the placeholder, uncomment this:
    # with pytest.raises(NotImplementedError, match="ENBS calculation is a placeholder."):
    #     enbs(sample_net_benefit_array_2strat, sample_trial_design_2arms, cost, dummy_model, sample_psa_2params)


def test_enbs_input_validation(
    sample_net_benefit_array_2strat: NetBenefitArray,
    sample_psa_2params: PSASample,
    sample_trial_design_2arms: TrialDesign,
):
    """Test ENBS input validation for incorrect types and negative cost."""
    pass

    def dummy_model(psa: PSASample) -> NetBenefitArray:
        return NetBenefitArray(values=np.array([[10, 12]]))

    with pytest.raises(
        InputError, match="'net_benefit_array' must be an instance of NetBenefitArray."
    ):
        enbs(None, sample_trial_design_2arms, 100.0, dummy_model, sample_psa_2params)  # type: ignore
    with pytest.raises(
        InputError, match="'trial_design' must be an instance of TrialDesign."
    ):
        enbs(
            sample_net_benefit_array_2strat,
            dummy_model,
            100.0,
            sample_psa_2params,
        )  # type: ignore
    with pytest.raises(
        InputError, match="'cost_of_study' must be a non-negative number."
    ):
        enbs(
            sample_net_benefit_array_2strat,
            sample_trial_design_2arms,
            -100.0,
            dummy_model,
            sample_psa_2params,
        )
    with pytest.raises(
        InputError, match="'model_function' must be a callable function."
    ):
        enbs(
            sample_net_benefit_array_2strat,
            sample_trial_design_2arms,
            "not_a_function",
            sample_psa_2params,
        )  # type: ignore
    with pytest.raises(
        InputError, match="'psa_sample' must be an instance of PSASample."
    ):
        enbs(
            sample_net_benefit_array_2strat,
            sample_trial_design_2arms,
            100.0,
            dummy_model,
        )  # type: ignore
