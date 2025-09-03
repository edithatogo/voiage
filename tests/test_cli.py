"""Test the CLI implementation."""

import subprocess
import sys
from pathlib import Path


def test_cli_evpi():
    """Test the EVPI CLI command."""
    # Get the path to the sample data
    sample_data_dir = Path(__file__).parent / "data"
    net_benefit_file = sample_data_dir / "sample_net_benefits.csv"
    
    # Run the CLI command
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evpi", 
        str(net_benefit_file),
        "--population", "100000",
        "--time-horizon", "10",
        "--discount-rate", "0.03"
    ], capture_output=True, text=True)
    
    # Check that the command executed successfully
    assert result.returncode == 0, f"Command failed with return code {result.returncode} and stderr: {result.stderr}"
    assert "EVPI:" in result.stdout
    print("EVPI CLI test passed!")


def test_cli_evppi():
    """Test the EVPPI CLI command."""
    # Get the path to the sample data
    sample_data_dir = Path(__file__).parent / "data"
    net_benefit_file = sample_data_dir / "sample_net_benefits.csv"
    parameter_file = sample_data_dir / "sample_parameters.csv"
    
    # Run the CLI command
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evppi",
        str(net_benefit_file),
        str(parameter_file),
        "--population", "100000",
        "--time-horizon", "10",
        "--discount-rate", "0.03"
    ], capture_output=True, text=True)
    
    # Check that the command executed successfully
    assert result.returncode == 0, f"Command failed with return code {result.returncode} and stderr: {result.stderr}"
    assert "EVPPI:" in result.stdout
    print("EVPPI CLI test passed!")


if __name__ == "__main__":
    test_cli_evpi()
    test_cli_evppi()
    print("All CLI tests passed!")