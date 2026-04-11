"""End-to-end tests for structural VOI CLI commands."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.e2e
class TestStructuralVOICLIE2E:
    """End-to-end tests for structural VOI CLI commands."""

    @pytest.fixture
    def structural_config_json(self, tmp_path):
        """Create a temporary structural VOI config JSON file."""
        config = {
            "structures": [
                {
                    "name": "model_a",
                    "probability": 0.6,
                    "net_benefits_file": "net_benefits_a.csv"
                },
                {
                    "name": "model_b",
                    "probability": 0.4,
                    "net_benefits_file": "net_benefits_b.csv"
                }
            ]
        }

        # Create net benefits CSV files
        np.random.seed(42)
        n_simulations = 100

        # Model A: Strategy 0 is better
        nb_a = np.random.rand(n_simulations, 2) * 100
        nb_a[:, 0] += 50  # Make strategy 0 better
        np.savetxt(
            tmp_path / "net_benefits_a.csv",
            nb_a,
            delimiter=",",
            header="Strategy_A,Strategy_B",
            comments=""
        )

        # Model B: Strategy 1 is better
        nb_b = np.random.rand(n_simulations, 2) * 100
        nb_b[:, 1] += 50  # Make strategy 1 better
        np.savetxt(
            tmp_path / "net_benefits_b.csv",
            nb_b,
            delimiter=",",
            header="Strategy_A,Strategy_B",
            comments=""
        )

        config_path = tmp_path / "structural_config.json"
        config_path.write_text(json.dumps(config))
        return config_path

    def test_cli_structural_evpi(self, structural_config_json):
        """Test structural EVPI calculation via CLI."""
        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-structural-evpi",
                str(structural_config_json)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Structural EVPI:" in result.stdout

        # Verify the EVPI value is a valid number
        evpi_value = float(result.stdout.strip().split(":")[1])
        assert evpi_value >= 0

    def test_cli_structural_evpi_with_population(self, structural_config_json):
        """Test structural EVPI with population scaling."""
        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-structural-evpi",
                str(structural_config_json),
                "--population", "10000",
                "--time-horizon", "10",
                "--discount-rate", "0.03"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Structural EVPI:" in result.stdout

        evpi_value = float(result.stdout.strip().split(":")[1])
        assert evpi_value >= 0

    def test_cli_structural_evpi_with_output(self, structural_config_json, tmp_path):
        """Test structural EVPI with output file."""
        output_file = tmp_path / "structural_evpi_result.txt"

        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-structural-evpi",
                str(structural_config_json),
                "--output", str(output_file)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "Structural EVPI:" in content

    def test_cli_structural_evppi(self, structural_config_json):
        """Test structural EVPPI calculation via CLI."""
        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-structural-evppi",
                str(structural_config_json),
                "--structures-of-interest", "0"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Structural EVPPI:" in result.stdout

        evppi_value = float(result.stdout.strip().split(":")[1])
        assert evppi_value >= 0

    def test_cli_structural_evppi_invalid(self, structural_config_json):
        """Test structural EVPPI with invalid structure index."""
        result = subprocess.run(
            [
                sys.executable, "-m", "voiage", "calculate-structural-evppi",
                str(structural_config_json),
                "--structures-of-interest", "99"
            ],
            capture_output=True,
            text=True
        )

        # Should fail with error
        assert result.returncode != 0
