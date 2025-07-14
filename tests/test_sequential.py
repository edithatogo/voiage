# tests/test_sequential.py

"""Unit tests for the sequential VOI methods in voiage.methods.sequential."""

import pytest
import numpy as np

from voiage.methods.sequential import sequential_voi
from voiage.exceptions import VoiageNotImplementedError
from voiage.schema import DynamicSpec, ParameterSet


def test_sequential_voi_placeholder():
    """Test that the sequential_voi function raises NotImplementedError."""

    def dummy_step_model(psa, action, dyn_spec):
        return {}

    dummy_psa = ParameterSet(parameters={"p": np.array([1])})
    dummy_dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    with pytest.raises(VoiageNotImplementedError):
        sequential_voi(dummy_step_model, dummy_psa, dummy_dyn_spec)
