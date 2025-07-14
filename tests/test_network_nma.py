# tests/test_network_nma.py

"""Unit tests for the network NMA VOI methods in voiage.methods.network_nma."""

import pytest
import numpy as np

from voiage.methods.network_nma import evsi_nma
from voiage.exceptions import VoiageNotImplementedError
from voiage.schema import ValueArray, ParameterSet, DecisionOption, TrialDesign


def test_evsi_nma_placeholder():
    """Test that the evsi_nma function raises NotImplementedError."""

    def dummy_nma_evaluator(psa, trial_design, data):
        return ValueArray(np.array([[0.0]]))

    dummy_psa = ParameterSet(parameters={"p": np.array([1])})
    dummy_trial = TrialDesign(arms=[DecisionOption(name="A", sample_size=10)])

    with pytest.raises(VoiageNotImplementedError):
        evsi_nma(dummy_nma_evaluator, dummy_psa, dummy_trial)
