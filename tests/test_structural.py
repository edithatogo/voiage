# tests/test_structural.py

"""Unit tests for the structural VOI methods in voiage.methods.structural."""

import pytest

from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.exceptions import VoiageNotImplementedError


def test_structural_evpi_placeholder():
    """Test that the structural_evpi function raises NotImplementedError."""
    with pytest.raises(VoiageNotImplementedError):
        structural_evpi([], [], [])


def test_structural_evppi_placeholder():
    """Test that the structural_evppi function raises NotImplementedError."""
    with pytest.raises(VoiageNotImplementedError):
        structural_evppi()
