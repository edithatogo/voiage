# tests/test_adaptive.py

"""Unit tests for the adaptive VOI methods in voiage.methods.adaptive."""

import pytest
import numpy as np

from voiage.methods.adaptive import adaptive_evsi
from voiage.exceptions import VoiageNotImplementedError
from voiage.schema import ValueArray, ParameterSet, DecisionOption, TrialDesign


def test_adaptive_evsi_placeholder():
    """Test that the adaptive_evsi function raises NotImplementedError."""

    def dummy_adaptive_sim(psa, design, rules):
        return ValueArray(np.array([[0.0]]))

    dummy_psa = ParameterSet(parameters={"p": np.array([1])})
    dummy_design = TrialDesign(arms=[DecisionOption(name="A", sample_size=10)])
    dummy_rules = {"stop_if_eff_at_interim1": 0.95}

    with pytest.raises(VoiageNotImplementedError):
        adaptive_evsi(dummy_adaptive_sim, dummy_psa, dummy_design, dummy_rules)
