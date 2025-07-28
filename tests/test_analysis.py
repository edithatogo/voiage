import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.core.data_structures import NetBenefitArray, PSASample
from voiage.methods.basic import evpi as func_evpi, evppi as func_evppi


def test_evpi_method_matches_function(sample_net_benefit_array_2strat: NetBenefitArray):
    analysis = DecisionAnalysis(sample_net_benefit_array_2strat)
    assert analysis.evpi() == func_evpi(sample_net_benefit_array_2strat)


def test_evppi_method_matches_function(evppi_test_data_simple):
    nb_values = evppi_test_data_simple["nb_values"]
    p_samples = evppi_test_data_simple["p_samples"]
    analysis = DecisionAnalysis(nb_values)
    val_method = analysis.evppi(p_samples)
    val_function = func_evppi(nb_values, p_samples)
    np.testing.assert_allclose(val_method, val_function)
