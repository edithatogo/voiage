# voiage/cli.py

"""
Command-Line Interface (CLI) for voiage.

This module provides CLI entry points for performing VOI analyses
or accessing voiage utilities from the command line.
It uses Typer for command-line argument parsing.
"""

import json
from pathlib import Path

import numpy as np
import typer

from voiage.core.io import read_parameter_set_csv, read_value_array_csv
from voiage.methods.basic import evpi, evppi
from voiage.methods.network_meta_analysis import (
    calculate_nma_evpi,
    calculate_nma_evppi,
)
from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.schema import ParameterSet

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
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted EVPI"
    ),
    discount_rate: float | None = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save EVPI result"
    ),
) -> None:
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
            with open(output_file, "w") as f:
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
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted EVPPI"
    ),
    discount_rate: float | None = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save EVPPI result"
    ),
) -> None:
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
            with open(output_file, "w") as f:
                f.write(result_str + "\n")
            typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_structural_evpi(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON config file defining model structures"
    ),
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted structural EVPI"
    ),
    discount_rate: float | None = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save structural EVPI result"
    ),
) -> None:
    r"""Calculate Structural Expected Value of Perfect Information (SEVPI).

    Structural EVPI quantifies the expected gain from knowing with certainty
    which model structure is the most appropriate one. The config file should
    be a JSON file with the following format:

    \b
    {
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
    """
    try:
        # Read config file
        with open(config_file) as f:
            config = json.load(f)

        if "structures" not in config:
            typer.echo("Error: Config file must contain 'structures' key", err=True)
            raise typer.Exit(code=1)

        structures = config["structures"]

        # Validate probabilities sum to 1
        total_prob = sum(s["probability"] for s in structures)
        if not np.isclose(total_prob, 1.0):
            typer.echo(f"Error: Structure probabilities must sum to 1 (got {total_prob})", err=True)
            raise typer.Exit(code=1)

        # Create model structure evaluators
        evaluators = []
        psa_samples = []

        for struct in structures:
            nb_file = struct["net_benefits_file"]
            nb_path = Path(nb_file)
            if not nb_path.is_absolute():
                nb_path = config_file.parent / nb_path

            # Read net benefits
            nba = read_value_array_csv(str(nb_path), skip_header=True)

            # Create evaluator function
            def make_evaluator(nb_array):
                def evaluator(psa_sample):
                    return nb_array
                return evaluator

            evaluators.append(make_evaluator(nba))

            # Create PSA sample (use net benefit values as proxy)
            psa = ParameterSet.from_numpy_or_dict({
                f"param_{i}": nba.numpy_values[:, i] if i < nba.numpy_values.shape[1] else nba.numpy_values[:, 0]
                for i in range(max(1, nba.numpy_values.shape[1]))
            })
            psa_samples.append(psa)

        # Extract probabilities
        probabilities = [s["probability"] for s in structures]

        # Calculate structural EVPI
        result = structural_evpi(
            evaluators,
            probabilities,
            psa_samples,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon
        )

        # Format result string
        result_str = f"Structural EVPI: {result:.6f}"

        # Print result to console
        typer.echo(result_str)

        # Save to output file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(result_str + "\n")
            typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError:
        typer.echo(f"Error: Config file not found at '{config_file}'", err=True)
        raise typer.Exit(code=1) from None
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in config file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_structural_evppi(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON config file defining model structures"
    ),
    structures_of_interest: list[int] = typer.Option(
        ...,
        "--structures-of-interest",
        "-s",
        help="Indices of structures to learn about (0-indexed, can specify multiple)"
    ),
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted structural EVPPI"
    ),
    discount_rate: float | None = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save structural EVPPI result"
    ),
) -> None:
    """Calculate Structural Expected Value of Partial Perfect Information (SEVPPI).

    Structural EVPPI quantifies the expected gain from resolving uncertainty
    about a specific subset of model structures.

    The config file format is the same as for structural EVPI.
    """
    try:
        if not structures_of_interest:
            typer.echo("Error: --structures-of-interest is required", err=True)
            raise typer.Exit(code=1)

        # Read config file
        with open(config_file) as f:
            config = json.load(f)

        if "structures" not in config:
            typer.echo("Error: Config file must contain 'structures' key", err=True)
            raise typer.Exit(code=1)

        structures = config["structures"]

        # Validate probabilities sum to 1
        total_prob = sum(s["probability"] for s in structures)
        if not np.isclose(total_prob, 1.0):
            typer.echo(f"Error: Structure probabilities must sum to 1 (got {total_prob})", err=True)
            raise typer.Exit(code=1)

        # Create model structure evaluators
        evaluators = []
        psa_samples = []

        for struct in structures:
            nb_file = struct["net_benefits_file"]
            nb_path = Path(nb_file)
            if not nb_path.is_absolute():
                nb_path = config_file.parent / nb_path

            # Read net benefits
            nba = read_value_array_csv(str(nb_path), skip_header=True)

            # Create evaluator function
            def make_evaluator(nb_array):
                def evaluator(psa_sample):
                    return nb_array
                return evaluator

            evaluators.append(make_evaluator(nba))

            # Create PSA sample
            psa = ParameterSet.from_numpy_or_dict({
                f"param_{i}": nba.numpy_values[:, i] if i < nba.numpy_values.shape[1] else nba.numpy_values[:, 0]
                for i in range(max(1, nba.numpy_values.shape[1]))
            })
            psa_samples.append(psa)

        # Extract probabilities
        probabilities = [s["probability"] for s in structures]

        # Calculate structural EVPPI
        result = structural_evppi(
            evaluators,
            probabilities,
            psa_samples,
            structures_of_interest=list(structures_of_interest),
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon
        )

        # Format result string
        result_str = f"Structural EVPPI: {result:.6f}"

        # Print result to console
        typer.echo(result_str)

        # Save to output file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(result_str + "\n")
            typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError:
        typer.echo(f"Error: Config file not found at '{config_file}'", err=True)
        raise typer.Exit(code=1) from None
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in config file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_nma_voi(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON config file defining NMA data"
    ),
    parameters_of_interest: str | None = typer.Option(
        None,
        "--parameters-of-interest",
        "-p",
        help="Comma-separated list of parameters for EVPPI (e.g., 'effect_A,effect_B')"
    ),
    willingness_to_pay: float | None = typer.Option(
        None,
        "--willingness-to-pay",
        "-w",
        help="Willingness-to-pay threshold per unit"
    ),
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted VOI"
    ),
    discount_rate: float | None = typer.Option(
        None,
        "--discount-rate",
        help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None,
        "--time-horizon",
        help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="File to save NMA VOI result"
    ),
) -> None:
    r"""Calculate VOI for Network Meta-Analysis (NMA-EVPI or NMA-EVPPI).

    The config file should be a JSON file with the following format:

    \b
    {
      "treatment_effects": {
        "Placebo-Drug_A": [0.5, 0.6, 0.4, ...],
        "Placebo-Drug_B": [0.7, 0.8, 0.6, ...],
        "Drug_A-Drug_B": [0.2, 0.1, 0.3, ...]
      },
      "n_studies": 10,
      "treatments": ["Placebo", "Drug_A", "Drug_B"],
      "outcome_type": "continuous"
    }

    If --parameters-of-interest is provided, calculates EVPPI; otherwise EVPI.
    """
    try:
        # Read config file
        with open(config_file) as f:
            config = json.load(f)

        # Convert list values to numpy arrays
        treatment_effects = {}
        for key, values in config.get("treatment_effects", {}).items():
            pair = tuple(key.split("-")) if isinstance(key, str) and "-" in key else key
            treatment_effects[pair] = np.asarray(values, dtype=float)

        config["treatment_effects"] = treatment_effects

        # Calculate NMA VOI
        if parameters_of_interest:
            # EVPPI calculation
            params_list = [p.strip() for p in parameters_of_interest.split(",")]

            # Generate parameter samples from treatment effects
            n_samples = next(iter(treatment_effects.values())).shape[0]
            parameter_samples = {
                param: np.random.rand(n_samples) for param in params_list
            }

            result = calculate_nma_evppi(
                config,
                parameters_of_interest=params_list,
                parameter_samples=parameter_samples,
                willingness_to_pay=willingness_to_pay,
                population=population,
                discount_rate=discount_rate,
                time_horizon=time_horizon,
            )
            result_str = f"NMA-EVPPI: {result:.6f}"
        else:
            # EVPI calculation
            result = calculate_nma_evpi(
                config,
                willingness_to_pay=willingness_to_pay,
                population=population,
                discount_rate=discount_rate,
                time_horizon=time_horizon,
            )
            result_str = f"NMA-EVPI: {result:.6f}"

        # Print result to console
        typer.echo(result_str)

        # Save to output file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(result_str + "\n")
            typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError:
        typer.echo(f"Error: Config file not found at '{config_file}'", err=True)
        raise typer.Exit(code=1) from None
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in config file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


# Add other commands for EVSI, ENBS, etc., as they become feasible for CLI.

# Example of how to run if using Typer directly (not through setup.py entry points)
if __name__ == "__main__":
    app()
