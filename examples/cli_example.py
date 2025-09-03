"""
Example script demonstrating CLI usage.

This script creates sample data files and shows how to run the CLI commands.
"""

import csv
import subprocess
import sys
from pathlib import Path


def create_sample_data():
    """Create sample data files for CLI demonstration."""
    # Create sample net benefits data
    net_benefits_data = [
        ['Strategy A', 'Strategy B', 'Strategy C'],
        [1000, 1200, 1100],
        [950, 1250, 1050],
        [1050, 1150, 1150],
        [900, 1300, 1000],
        [1100, 1100, 1200],
        [980, 1220, 1080],
        [1020, 1180, 1120],
        [960, 1260, 1040],
        [1040, 1160, 1160],
        [920, 1280, 1020]
    ]
    
    with open('example_net_benefits.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(net_benefits_data)
    
    # Create sample parameters data
    parameters_data = [
        ['effectiveness', 'cost'],
        [0.5, 0.3],
        [0.6, 0.4],
        [0.4, 0.2],
        [0.7, 0.5],
        [0.5, 0.3],
        [0.5, 0.3],
        [0.6, 0.4],
        [0.4, 0.2],
        [0.7, 0.5],
        [0.5, 0.3]
    ]
    
    with open('example_parameters.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(parameters_data)
    
    print("Sample data files created:")
    print("- example_net_benefits.csv")
    print("- example_parameters.csv")


def run_evpi_example():
    """Run EVPI calculation example."""
    print("\n--- Running EVPI Calculation ---")
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evpi",
        "example_net_benefits.csv",
        "--population", "100000",
        "--time-horizon", "10",
        "--discount-rate", "0.03"
    ], capture_output=True, text=True)
    
    print("Command output:")
    print(result.stdout)
    if result.stderr:
        print("Command errors:")
        print(result.stderr)


def run_evppi_example():
    """Run EVPPI calculation example."""
    print("\n--- Running EVPPI Calculation ---")
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evppi",
        "example_net_benefits.csv",
        "example_parameters.csv",
        "--population", "100000",
        "--time-horizon", "10",
        "--discount-rate", "0.03"
    ], capture_output=True, text=True)
    
    print("Command output:")
    print(result.stdout)
    if result.stderr:
        print("Command errors:")
        print(result.stderr)


def run_with_output_example():
    """Run calculation with output file example."""
    print("\n--- Running Calculation with Output File ---")
    result = subprocess.run([
        sys.executable, "-m", "voiage.cli", "calculate-evpi",
        "example_net_benefits.csv",
        "--population", "100000",
        "--time-horizon", "10",
        "--discount-rate", "0.03",
        "--output", "evpi_result.txt"
    ], capture_output=True, text=True)
    
    print("Command output:")
    print(result.stdout)
    if result.stderr:
        print("Command errors:")
        print(result.stderr)
    
    # Show the output file content
    if Path("evpi_result.txt").exists():
        print("\nOutput file content:")
        with open("evpi_result.txt", "r") as f:
            print(f.read())


if __name__ == "__main__":
    print("voiage CLI Usage Examples")
    print("========================")
    
    # Create sample data
    create_sample_data()
    
    # Run examples
    run_evpi_example()
    run_evppi_example()
    run_with_output_example()
    
    print("\nExamples completed!")