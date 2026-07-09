"""
Example script demonstrating CLI usage.

This script uses the committed sample files and shows how to run the CLI
commands without writing generated inputs into the repository root.
"""

from pathlib import Path
from shutil import which
import subprocess
import sys
from tempfile import TemporaryDirectory


SAMPLE_DIR = Path(__file__).resolve().parent / "cli_samples"
NET_BENEFIT_FILE = SAMPLE_DIR / "evpi_net_benefit.csv"
PARAMETER_FILE = SAMPLE_DIR / "evppi_parameters.csv"


def _cli_command(args: list[str]) -> list[str]:
    executable = which("voiage")
    if executable is not None:
        return [executable, *args]
    return [sys.executable, "-m", "voiage.cli", *args]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a CLI command and fail fast if the example stops matching reality."""
    result = subprocess.run(_cli_command(args), capture_output=True, text=True)

    print("Command output:")
    print(result.stdout)
    if result.stderr:
        print("Command errors:")
        print(result.stderr)
    result.check_returncode()
    return result


def run_evpi_example():
    """Run EVPI calculation example."""
    print("\n--- Running EVPI Calculation ---")
    _run_cli([
        "calculate-evpi",
        str(NET_BENEFIT_FILE),
        "--population",
        "100000",
        "--time-horizon",
        "10",
        "--discount-rate",
        "0.03",
    ])


def run_evppi_example():
    """Run EVPPI calculation example."""
    print("\n--- Running EVPPI Calculation ---")
    _run_cli([
        "calculate-evppi",
        str(NET_BENEFIT_FILE),
        str(PARAMETER_FILE),
        "--population",
        "100000",
        "--time-horizon",
        "10",
        "--discount-rate",
        "0.03",
    ])


def run_with_output_example():
    """Run calculation with output file example."""
    print("\n--- Running Calculation with Output File ---")
    with TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "evpi_result.txt"
        _run_cli([
            "calculate-evpi",
            str(NET_BENEFIT_FILE),
            "--population",
            "100000",
            "--time-horizon",
            "10",
            "--discount-rate",
            "0.03",
            "--output",
            str(output_file),
        ])

        if output_file.exists():
            print("\nOutput file content:")
            print(output_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("voiage CLI Usage Examples")
    print("========================")

    # Run examples
    run_evpi_example()
    run_evppi_example()
    run_with_output_example()

    print("\nExamples completed!")
