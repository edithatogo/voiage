# tests/test_calibration.py

"""Unit tests for the calibration VOI methods in voiage.methods.calibration."""

import pytest
import numpy as np

from voiage.methods.calibration import voi_calibration
from voiage.exceptions import VoiageNotImplementedError
from voiage.schema import ValueArray, ParameterSet


def test_voi_calibration_placeholder():
    """Test that the voi_calibration function raises NotImplementedError."""

    def dummy_cal_modeler(psa, design, spec):
        return ValueArray(np.array([[0.0]]))

    dummy_psa = ParameterSet(parameters={"p": np.array([1])})
    dummy_design = {"experiment_type": "lab", "n_runs": 10}
    dummy_spec = {"method": "bayesian_history_matching"}

    with pytest.raises(VoiageNotImplementedError):
        voi_calibration(dummy_cal_modeler, dummy_psa, dummy_design, dummy_spec)
