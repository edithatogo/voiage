# voiage/cli.py

"""
Command-Line Interface (CLI) for voiage.

This module provides CLI entry points for performing VOI analyses
or accessing voiage utilities from the command line.
It uses Typer for command-line argument parsing.
"""

from pathlib import Path
from typing import Optional

import typer

from voiage.core.io import read_parameter_set_csv, read_value_array_csv
from voiage.methods.basic import evpi, evppi

app = typer.Typer(help="voiage: A Command-Line Interface for Value of Information Analysis.")


@app.command()
def calculate_evpi(
    net_benefit_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV containing net benefits (samples x strategies)"
    ),
    population: Optional[float] = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted EVPI"
    ),
    discount_rate: Optional[float] = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: Optional[float] = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save EVPI result"
    ),
):
    """Calculate Expected Value of Perfect Information (EVPI) from input data."""
    try:
        # Read net benefit data from CSV
        nba = read_value_array_csv(str(net_benefit_file), skip_header=True)

        # Calculate EVPI
        result = evpi(
            nba,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon
        )

        # Format result string
        result_str = f"EVPI: {result:.6f}"

        # Print result to console
        typer.echo(result_str)

        # Save to output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result_str + "\n")
            typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError:
        typer.echo(f"Error: Net benefit file not found at '{net_benefit_file}'", err=True)
        raise typer.Exit(code=1) from None
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_evppi(
    net_benefit_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV containing net benefits (samples x strategies)"
    ),
    parameter_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV for parameters of interest (samples x params)"
    ),
    population: Optional[float] = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted EVPPI"
    ),
    discount_rate: Optional[float] = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: Optional[float] = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save EVPPI result"
    ),
):
    """Calculate Expected Value of Partial Perfect Information (EVPPI)."""
    try:
        # Read net benefit data from CSV
        nba = read_value_array_csv(str(net_benefit_file), skip_header=True)

        # Read parameter data from CSV
        param_set = read_parameter_set_csv(str(parameter_file), skip_header=True)

        # Get parameter names for EVPPI calculation
        parameter_names = param_set.parameter_names

        # Calculate EVPPI
        result = evppi(
            nb_array=nba,
            parameter_samples=param_set,
            parameters_of_interest=parameter_names,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon
        )

        # Format result string
        result_str = f"EVPPI: {result:.6f}"

        # Print result to console
        typer.echo(result_str)

        # Save to output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result_str + "\n")
            typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


# Add other commands for EVSI, ENBS, etc., as they become feasible for CLI.

# Example of how to run if using Typer directly (not through setup.py entry points)
if __name__ == "__main__":
    app()
