# tests/test_structural.py

"""Unit tests for the structural VOI methods in pyvoi.methods.structural."""

import numpy as np
import pytest

from pyvoi.core.data_structures import NetBenefitArray, PSASample
from pyvoi.exceptions import InputError
from pyvoi.methods.structural import structural_evpi

# --- Fixtures for Structural EVPI ---


@pytest.fixture()
def structural_evpi_test_data():
    """Provide a standard set of data for testing structural_evpi."""
    # Two model structures
    # Structure 1: Strat A is slightly better on average
    # Structure 2: Strat B is clearly better on average

    # PSA samples for Structure 1
    psa1_params = {
        "p1": np.array([0.5, 0.6, 0.7]),
        "p2": np.array([10, 11, 9]),
    }
    psa1 = PSASample(parameters=psa1_params)

    # PSA samples for Structure 2
    psa2_params = {
        "q1": np.array([100, 110, 120]),
        "q2": np.array([0.8, 0.9, 0.7]),
    }
    psa2 = PSASample(parameters=psa2_params)

    # Model Evaluator for Structure 1
    def model1_evaluator(psa: PSASample) -> NetBenefitArray:
        nb_a = (
            psa.parameters["p1"] * 20 + psa.parameters["p2"]
        )  # ~10-14 + 9-11 = ~20-25
        nb_b = (
            psa.parameters["p1"] * 10 + psa.parameters["p2"] * 1.5
        )  # ~5-7 + 13-16 = ~18-23
        return NetBenefitArray(values=np.stack([nb_a, nb_b], axis=1))

    # Model Evaluator for Structure 2
    def model2_evaluator(psa: PSASample) -> NetBenefitArray:
        nb_a = psa.parameters["q1"] * psa.parameters["q2"]  # ~80-108
        nb_b = psa.parameters["q1"] / psa.parameters["q2"] * 2  # ~250-340
        return NetBenefitArray(values=np.stack([nb_a, nb_b], axis=1))

    return {
        "evaluators": [model1_evaluator, model2_evaluator],
        "psa_samples": [psa1, psa2],
        "probabilities": [0.5, 0.5],
    }


# --- Tests for structural_evpi ---


def test_structural_evpi_calculation(structural_evpi_test_data):
    """Test the main calculation of structural_evpi."""
    data = structural_evpi_test_data
    sevpi = structural_evpi(
        model_structure_evaluators=data["evaluators"],
        structure_probabilities=data["probabilities"],
        psa_samples_per_structure=data["psa_samples"],
    )

    # Manual calculation for verification:
    # --- Structure 1 ---
    # NB_A1 = [0.5*20+10, 0.6*20+11, 0.7*20+9] = [20, 23, 23] -> E[NB_A1] = 22
    # NB_B1 = [0.5*10+15, 0.6*10+16.5, 0.7*10+13.5] = [20, 22.5, 20.5] -> E[NB_B1] = 21
    # E_theta|S1[NB(d,S1)] = [22, 21]
    # max_nb_per_sample_S1 = [max(20,20), max(23,22.5), max(23,20.5)] = [20, 23, 23]
    # E_theta|S1[max_d NB(d,S1)] = mean([20, 23, 23]) = 22

    # --- Structure 2 ---
    # NB_A2 = [80, 99, 84] -> E[NB_A2] = 87.666
    # NB_B2 = [250, 244.44, 342.85] -> E[NB_B2] = 278.96
    # E_theta|S2[NB(d,S2)] = [87.666, 278.96]
    # max_nb_per_sample_S2 = [250, 244.44, 342.85]
    # E_theta|S2[max_d NB(d,S2)] = mean([250, 244.44, 342.85]) = 278.96

    # --- Combine ---
    # Term 1 = P(S1)*E[max_d NB|S1] + P(S2)*E[max_d NB|S2]
    #        = 0.5 * 22 + 0.5 * 278.96 = 11 + 139.48 = 150.48
    # Term 2 = max_d ( P(S1)*E[NB_d|S1] + P(S2)*E[NB_d|S2] )
    # E[NB_A] = 0.5*22 + 0.5*87.666 = 11 + 43.833 = 54.833
    # E[NB_B] = 0.5*21 + 0.5*278.96 = 10.5 + 139.48 = 149.98
    # max(54.833, 149.98) = 149.98
    # SEVPI = 150.48 - 149.98 = 0.5
    expected_sevpi = 0.5
    if not np.isclose(sevpi, expected_sevpi, atol=1e-2):
        raise ValueError("SEVPI calculation failed.")


def test_structural_evpi_no_uncertainty():
    """Test SEVPI when there is no structural uncertainty (one structure)."""
    psa = PSASample(parameters={"p": np.array([1, 2, 3])})

    def model_eval(p: PSASample) -> NetBenefitArray:
        return NetBenefitArray(values=np.array([[10, 20], [10, 20], [10, 20]]))

    sevpi = structural_evpi(
        model_structure_evaluators=[model_eval],
        structure_probabilities=[1.0],
        psa_samples_per_structure=[psa],
    )
    if not np.isclose(sevpi, 0.0):
        raise ValueError("SEVPI with no structural uncertainty should be 0.")


def test_structural_evpi_population_scaling(structural_evpi_test_data):
    """Test SEVPI with population scaling."""
    data = structural_evpi_test_data
    per_decision_sevpi = 0.5  # From the main test
    population = 1000
    time_horizon = 10
    discount_rate = 0.03

    annuity = (1 - (1 + discount_rate) ** (-time_horizon)) / discount_rate
    expected_pop_sevpi = per_decision_sevpi * population * annuity

    calculated_pop_sevpi = structural_evpi(
        model_structure_evaluators=data["evaluators"],
        structure_probabilities=data["probabilities"],
        psa_samples_per_structure=data["psa_samples"],
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )
    if not np.isclose(calculated_pop_sevpi, expected_pop_sevpi):
        raise ValueError("Population SEVPI calculation failed.")


def test_structural_evpi_invalid_inputs(structural_evpi_test_data):
    """Test SEVPI with various invalid inputs."""
    data = structural_evpi_test_data

    # Mismatched lengths
    with pytest.raises(InputError, match="must have the same length"):
        structural_evpi(data["evaluators"], [0.5], data["psa_samples"])

    # Probabilities don't sum to 1
    with pytest.raises(InputError, match="must sum to 1"):
        structural_evpi(data["evaluators"], [0.4, 0.4], data["psa_samples"])

    # Inconsistent number of strategies
    def model3_evaluator(psa: PSASample) -> NetBenefitArray:
        return NetBenefitArray(values=np.array([[1, 2, 3]]))

    with pytest.raises(
        InputError, match="must evaluate the same number of decision strategies"
    ):
        structural_evpi(
            [data["evaluators"][0], model3_evaluator],
            [0.5, 0.5],
            data["psa_samples"],
        )
