# tests/test_cli.py

"""Unit tests for the CLI in voiage.cli."""

import pytest

from voiage.cli import calculate_evpi_cli, calculate_evppi_cli
from voiage.exceptions import VoiageNotImplementedError


def test_calculate_evpi_cli_placeholder():
    """Test that the EVPI CLI function raises NotImplementedError."""
    with pytest.raises(VoiageNotImplementedError):
        calculate_evpi_cli("dummy_nb.csv")


def test_calculate_evppi_cli_placeholder():
    """Test that the EVPPI CLI function raises NotImplementedError."""
    with pytest.raises(VoiageNotImplementedError):
        calculate_evppi_cli("dummy_nb.csv", "dummy_params.csv")
