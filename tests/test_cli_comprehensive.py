"""Comprehensive tests for the CLI implementation."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile


def test_evpi_cli():
    """Test the EVPI CLI command with various options."""
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create sample net benefits data
        net_benefits_file = tmpdir_path / "net_benefits.csv"
        net_benefits_data = [
            ['Strategy A', 'Strategy B'],
            [1000, 1200],
            [950, 1250],
            [1050, 1150],
            [900, 1300],
            [1100, 1100]
        ]

        with open(net_benefits_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(net_benefits_data)

        # Test basic EVPI calculation
        result = subprocess.run([
            sys.executable, "-m", "voiage.cli", "calculate-evpi",
            str(net_benefits_file)
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "EVPI:" in result.stdout
        evpi_value = float(result.stdout.split(":")[1].strip())
        assert evpi_value >= 0

        # Test EVPI with population scaling
        result = subprocess.run([
            sys.executable, "-m", "voiage.cli", "calculate-evpi",
            str(net_benefits_file),
            "--population", "100000",
            "--time-horizon", "10",
            "--discount-rate", "0.03"
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "EVPI:" in result.stdout
        scaled_evpi_value = float(result.stdout.split(":")[1].strip())
        assert scaled_evpi_value >= 0
        # Scaled EVPI should be larger than unscaled
        assert scaled_evpi_value >= evpi_value

        # Test EVPI with output file
        output_file = tmpdir_path / "evpi_result.txt"
        result = subprocess.run([
            sys.executable, "-m", "voiage.cli", "calculate-evpi",
            str(net_benefits_file),
            "--output", str(output_file)
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "EVPI:" in result.stdout
        assert "Result saved to" in result.stdout
        assert output_file.exists()

        # Check output file content
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert "EVPI:" in content


def test_evppi_cli():
    """Test the EVPPI CLI command with various options."""
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create sample net benefits data
        net_benefits_file = tmpdir_path / "net_benefits.csv"
        net_benefits_data = [
            ['Strategy A', 'Strategy B'],
            [1000, 1200],
            [950, 1250],
            [1050, 1150],
            [900, 1300],
            [1100, 1100]
        ]

        with open(net_benefits_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(net_benefits_data)

        # Create sample parameters data
        parameters_file = tmpdir_path / "parameters.csv"
        parameters_data = [
            ['param1', 'param2'],
            [0.5, 0.3],
            [0.6, 0.4],
            [0.4, 0.2],
            [0.7, 0.5],
            [0.5, 0.3]
        ]

        with open(parameters_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(parameters_data)

        # Test basic EVPPI calculation
        result = subprocess.run([
            sys.executable, "-m", "voiage.cli", "calculate-evppi",
            str(net_benefits_file),
            str(parameters_file)
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout
        evppi_value = float(result.stdout.split(":")[1].strip())
        assert evppi_value >= 0

        # Test EVPPI with population scaling
        result = subprocess.run([
            sys.executable, "-m", "voiage.cli", "calculate-evppi",
            str(net_benefits_file),
            str(parameters_file),
            "--population", "100000",
            "--time-horizon", "10",
            "--discount-rate", "0.03"
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout
        scaled_evppi_value = float(result.stdout.split(":")[1].strip())
        assert scaled_evppi_value >= 0

        # Test EVPPI with output file
        output_file = tmpdir_path / "evppi_result.txt"
        result = subprocess.run([
            sys.executable, "-m", "voiage.cli", "calculate-evppi",
            str(net_benefits_file),
            str(parameters_file),
            "--output", str(output_file)
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "EVPPI:" in result.stdout
        assert "Result saved to" in result.stdout
        assert output_file.exists()

        # Check output file content
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert "EVPPI:" in content


def test_cli_help():
    """Test CLI help commands."""
    # Test main help
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "--help"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "voiage" in result.stdout
    assert "calculate-evpi" in result.stdout
    assert "calculate-evppi" in result.stdout

    # Test calculate-evpi help
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evpi", "--help"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "calculate-evpi" in result.stdout
    assert "NET_BENEFIT_FILE" in result.stdout

    # Test calculate-evppi help
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evppi", "--help"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "calculate-evppi" in result.stdout
    assert "NET_BENEFIT_FILE" in result.stdout
    assert "PARAMETER_FILE" in result.stdout


def test_cli_error_handling():
    """Test CLI error handling."""
    # Test missing file
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evpi",
        "nonexistent_file.csv"
    ], capture_output=True, text=True)

    # This should fail because the file doesn't exist
    assert result.returncode != 0
    assert "Error" in result.stderr or "error" in result.stderr.lower()


if __name__ == "__main__":
    test_evpi_cli()
    test_evppi_cli()
    test_cli_help()
    test_cli_error_handling()
    print("All CLI comprehensive tests passed!")
