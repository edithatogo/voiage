# tests/test_observational.py

"""Unit tests for the observational VOI methods in voiage.methods.observational."""

import pytest
import numpy as np

from voiage.methods.observational import voi_observational
from voiage.exceptions import VoiageNotImplementedError
from voiage.schema import ValueArray, ParameterSet


def test_voi_observational_placeholder():
    """Test that the voi_observational function raises NotImplementedError."""

    def dummy_obs_modeler(psa, design, biases):
        return ValueArray(np.array([[0.0]]))

    dummy_psa = ParameterSet(parameters={"p": np.array([1])})
    dummy_design = {"type": "cohort", "size": 1000}
    dummy_biases = {"confounding_strength": 0.2}

    with pytest.raises(VoiageNotImplementedError):
        voi_observational(dummy_obs_modeler, dummy_psa, dummy_design, dummy_biases)
