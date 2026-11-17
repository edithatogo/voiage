"""Checks for the stated limits of the observational validation example."""

import importlib
import sys

import numpy as np
import pytest

from examples.observational_validation_example import (
    sophisticated_observational_modeler,
)
from voiage.schema import ParameterSet


def test_importing_example_preserves_module_search_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing an example must not shadow an installed package under test."""
    original_path = sys.path.copy()
    monkeypatch.setattr(sys, "path", original_path.copy())
    importlib.reload(sys.modules[sophisticated_observational_modeler.__module__])
    assert sys.path == original_path


def test_sophisticated_example_rejects_unmodelled_selection_bias() -> None:
    """A supplied nonzero selection bias must not be silently ignored."""
    samples = ParameterSet.from_numpy_or_dict(
        {
            "effectiveness": np.array([0.6, 0.7]),
            "cost": np.array([5_000.0, 5_500.0]),
            "utility": np.array([0.8, 0.9]),
        }
    )

    with pytest.raises(ValueError, match="does not model selection bias"):
        sophisticated_observational_modeler(
            samples, {}, {"selection_bias": {"probability": 0.1}}
        )

    result = sophisticated_observational_modeler(
        samples, {}, {"selection_bias": {"probability": 0.0}}
    )
    assert result.numpy_values.shape == (2, 2)
