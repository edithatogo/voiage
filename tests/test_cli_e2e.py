"""
End-to-end tests for voiage CLI.

These tests run the actual CLI commands and verify the output.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.e2e
class TestCLIE2E:
    """End-to-end tests for voiage CLI."""

    @pytest.fixture
    def net_benefits_csv(self, tmp_path):
        """Create a temporary net benefits CSV file."""
        np.random.seed(42)
        n_simulations = 100
        data = {
            'Standard_Care': np.random.rand(n_simulations) * 100 + 50,
            'Treatment_A': np.random.rand(n_simulations) * 120 + 60,
            'Treatment_B': np.random.rand(n_simulations) * 110 + 55,
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "net_benefits.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def parameters_csv(self, tmp_path):
        """Create a temporary parameters CSV file."""
        np.random.seed(42)
        n_simulations = 100
        data = {
            'effectiveness': np.random.rand(n_simulations),
            'cost_multiplier': np.random.rand(n_simulations) * 2,
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "parameters.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_cli_evpi(self, net_benefits_csv):
        """Test EVPI calculation via CLI."""
        result = subprocess.run(
            [sys.executable, "-m", "voiage", "calculate-evpi", str(net_benefits_csv)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "EVPI:" in result.stdout

        # Verify the EVPI value is a valid number
        evpi_value = float(result.stdout.strip().split(":")[1])
        assert evpi_value >= 0

    def test_cli_evpi_with_output(self, net_benefits_csv, tmp_path):
        """Test EVPI calculation with output file."""
        output_file = tmp_path / "evpi_result.txt"

        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-evpi",
                str(net_benefits_csv),
                "--output", str(output_file)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Verify output file contains the result
        content = output_file.read_text()
        assert "EVPI:" in content

    def test_cli_evppi(self, net_benefits_csv, parameters_csv):
        """Test EVPPI calculation via CLI."""
        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-evppi",
                str(net_benefits_csv),
                str(parameters_csv)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout

    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "voiage", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "voiage" in result.stdout.lower()
