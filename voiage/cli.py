# voiage/cli.py

"""
Command-Line Interface (CLI) for voiage.

This module provides CLI entry points for performing VOI analyses
or accessing voiage utilities from the command line.
It uses Typer for command-line argument parsing.
"""

# Typer argument declarations intentionally call helper constructors in defaults.
# ruff: noqa: B008, TRY301

from collections.abc import Callable, Sequence
import csv
import io
import json
import logging
from pathlib import Path
import re
import sys
from typing import Literal, cast

import numpy as np
import typer

from voiage.core.io import (
    import_callable,
    read_parameter_set_csv,
    read_value_array_csv,
)
from voiage.methods.adaptive import (
    adaptive_evsi,
    bayesian_adaptive_trial_simulator,
    sophisticated_adaptive_trial_simulator,
)
from voiage.methods.basic import evpi, evppi
from voiage.methods.ceaf import calculate_ceaf as calculate_ceaf_result
from voiage.methods.dominance import calculate_dominance as calculate_dominance_result
from voiage.methods.network_meta_analysis import (
    calculate_nma_evpi,
    calculate_nma_evppi,
)
from voiage.methods.perspective import (
    ValueOfPerspectiveResult,
)
from voiage.methods.perspective import (
    value_of_perspective as calculate_perspective_result,
)
from voiage.methods.portfolio import portfolio_voi
from voiage.methods.sample_information import enbs, evsi
from voiage.methods.sequential import sequential_voi
from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.plot.ceac import plot_ceac as render_ceac
from voiage.plot.ceaf import plot_ceaf as render_ceaf
from voiage.plot.dominance import plot_cost_effectiveness_plane as render_dominance
from voiage.plot.perspective import plot_perspective_regret as render_perspective_regret
from voiage.plot.voi_curves import (
    plot_evpi_vs_wtp as render_evpi_vs_wtp,
)
from voiage.plot.voi_curves import (
    plot_evsi_vs_sample_size as render_evsi_vs_sample_size,
)
from voiage.reporting import build_cheers_reporting
from voiage.schema import (
    DynamicSpec,
    ParameterSet,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
    ValueArray,
)

app = typer.Typer(
    help="voiage: A Command-Line Interface for Value of Information Analysis."
)

OutputFormat = Literal["text", "json", "csv"]

_SCALAR_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_CLI_LOGGER = logging.getLogger("voiage.cli")
_CLI_STATE: dict[str, bool | OutputFormat] = {
    "output_format": "text",
    "quiet": False,
    "verbose": False,
}

_CONFIG_TEMPLATES: dict[str, dict[str, object]] = {
    "evsi": {
        "command": "calculate-evsi",
        "description": "Template for EVSI analysis inputs.",
        "parameter_file": "parameters.csv",
        "trial_design_file": "trial_design.json",
        "model": "your.module:callable",
        "net_benefit_file": None,
        "method": "two_loop",
        "metamodel": "linear",
        "n_outer_loops": 100,
        "n_inner_loops": 1000,
        "population": None,
        "discount_rate": None,
        "time_horizon": None,
    },
    "evppi": {
        "command": "calculate-evppi",
        "description": "Template for EVPPI analysis inputs.",
        "net_benefit_file": "net_benefits.csv",
        "parameter_file": "parameters.csv",
        "population": None,
        "discount_rate": None,
        "time_horizon": None,
    },
    "nma-voi": {
        "command": "calculate-nma-voi",
        "description": "Template for NMA VOI inputs.",
        "treatment_effects": {"Placebo-Drug_A": [0.0, 0.0]},
        "n_studies": 0,
        "treatments": ["Placebo", "Drug_A"],
        "outcome_type": "continuous",
        "parameters_of_interest": None,
        "willingness_to_pay": None,
        "population": None,
        "discount_rate": None,
        "time_horizon": None,
    },
    "perspective": {
        "command": "calculate-perspective",
        "description": "Template for experimental Value of Perspective inputs.",
        "net_benefit": [
            [[10.0, 7.0], [8.0, 11.0]],
            [[10.0, 7.0], [8.0, 11.0]],
        ],
        "strategy_names": ["A", "B"],
        "perspective_names": ["payer", "societal"],
        "perspective_weights": {"payer": 0.5, "societal": 0.5},
        "reference_perspective": "payer",
    },
}


def _configure_cli_logging(verbose: bool) -> None:
    """Configure the CLI-local logger for the current invocation."""
    _CLI_STATE["verbose"] = verbose

    handler = next(
        (
            existing_handler
            for existing_handler in _CLI_LOGGER.handlers
            if getattr(existing_handler, "_voiage_cli_handler", False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        handler._voiage_cli_handler = True  # type: ignore[attr-defined]
        _CLI_LOGGER.addHandler(handler)

    handler.stream = sys.stderr
    level = logging.DEBUG if verbose else logging.ERROR
    handler.setLevel(level)
    _CLI_LOGGER.setLevel(level)
    _CLI_LOGGER.propagate = False


def _log_cli_debug(command: str, **details: object) -> None:
    """Emit a structured debug message when verbose logging is enabled."""
    if not cast("bool", _CLI_STATE["verbose"]):
        return

    if details:
        formatted_details = ", ".join(
            f"{key}={value!r}" for key, value in details.items()
        )
        _CLI_LOGGER.debug("%s: %s", command, formatted_details)
        return

    _CLI_LOGGER.debug("%s", command)


@app.callback(invoke_without_command=False)
def main(
    output_format: OutputFormat = typer.Option(
        "text",
        "--format",
        help="Output format for command results: text, json, or csv",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress confirmation messages and keep only the result output",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit debug diagnostics to stderr without changing stdout output",
    ),
) -> None:
    """Configure CLI-wide output formatting."""
    _CLI_STATE["output_format"] = output_format
    _CLI_STATE["quiet"] = quiet
    _configure_cli_logging(verbose)


def _read_trial_design_json(path: Path) -> TrialDesign:
    """Read a trial design JSON file."""
    return TrialDesign.from_dict(_read_json_file(path))


def _read_json_file(path: Path) -> object:
    """Read a JSON file and return the decoded object."""
    _log_cli_debug("read-json", path=str(path))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_figure(ax: object, output_file: Path) -> None:
    """Save a Matplotlib axes figure to disk."""
    _log_cli_debug("save-figure", output_file=str(output_file))
    figure = getattr(ax, "figure", None)
    if figure is None:
        raise TypeError("Plot function did not return a Matplotlib axes object.")
    figure.savefig(output_file, bbox_inches="tight")


def _csv_string(value: object) -> str:
    """Convert a value to a CSV-safe string."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _format_output(text_output: str, payload: dict[str, object]) -> str:
    """Format a command response according to the active CLI output format."""
    output_format = cast("OutputFormat", _CLI_STATE["output_format"])
    if output_format == "text":
        return text_output
    if output_format == "json":
        return json.dumps(payload, indent=2, sort_keys=True)
    if output_format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(payload))
        writer.writeheader()
        writer.writerow({key: _csv_string(value) for key, value in payload.items()})
        return buffer.getvalue().strip()
    raise ValueError(f"Unsupported output format: {output_format}")


def _should_echo_status_messages() -> bool:
    """Return whether non-result confirmation messages should be emitted."""
    return (
        cast("bool", _CLI_STATE["quiet"]) is False
        and cast("OutputFormat", _CLI_STATE["output_format"]) == "text"
    )


def _write_output_file(output_file: Path, content: str) -> None:
    """Write command output to a file with a trailing newline."""
    _log_cli_debug("write-output", output_file=str(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content + "\n")


def _read_plot_surface(path: Path) -> tuple[ValueArray, list[float]]:
    """Read a 3D net-benefit surface from JSON for CEAC/CEAF plotting."""
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Plot surface file must contain a JSON object.")
    payload = cast("dict[str, object]", payload)
    if "net_benefit" not in payload or "wtp_thresholds" not in payload:
        raise TypeError(
            "Plot surface file must contain 'net_benefit' and 'wtp_thresholds'."
        )

    net_benefit = np.asarray(
        cast("list[list[list[float]]]", payload["net_benefit"]),
        dtype=float,
    )
    if net_benefit.ndim != 3:
        raise TypeError("`net_benefit` must be a 3D array.")

    strategy_names = payload.get("strategy_names")
    if strategy_names is None:
        strategy_names = [f"Strategy {idx + 1}" for idx in range(net_benefit.shape[1])]
    if not isinstance(strategy_names, list):
        raise TypeError("'strategy_names' must be a list when provided.")

    wtp_thresholds = payload["wtp_thresholds"]
    if not isinstance(wtp_thresholds, list):
        raise TypeError("'wtp_thresholds' must be a list.")

    import xarray as xr

    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies", "n_wtp"), net_benefit)},
        coords={
            "n_samples": np.arange(net_benefit.shape[0]),
            "n_strategies": np.arange(net_benefit.shape[1]),
            "n_wtp": np.arange(net_benefit.shape[2]),
            "strategy": ("n_strategies", strategy_names),
        },
    )
    return ValueArray(dataset=dataset), [
        float(cast("float", value)) for value in wtp_thresholds
    ]


def _read_perspective_surface(
    path: Path,
) -> tuple[
    ValueArray,
    list[str] | None,
    list[str] | None,
    object,
    object,
]:
    """Read a multi-perspective net-benefit surface from JSON."""
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Perspective surface file must contain a JSON object.")
    payload = cast("dict[str, object]", payload)
    if "net_benefit" not in payload:
        raise TypeError("Perspective surface file must contain 'net_benefit'.")

    net_benefit = np.asarray(
        cast("list[list[list[float]]]", payload["net_benefit"]),
        dtype=float,
    )
    if net_benefit.ndim != 3:
        raise TypeError(
            "`net_benefit` must be a 3D array (samples x strategies x perspectives)."
        )

    strategy_names = payload.get("strategy_names")
    if strategy_names is not None and not isinstance(strategy_names, list):
        raise TypeError("'strategy_names' must be a list when provided.")
    perspective_names = payload.get("perspective_names")
    if perspective_names is not None and not isinstance(perspective_names, list):
        raise TypeError("'perspective_names' must be a list when provided.")

    value_array = ValueArray.from_numpy_perspectives(
        net_benefit,
        strategy_names=cast("list[str] | None", strategy_names),
        perspective_names=cast("list[str] | None", perspective_names),
    )
    return (
        value_array,
        cast("list[str] | None", strategy_names),
        cast("list[str] | None", perspective_names),
        payload.get("perspective_weights"),
        payload.get("reference_perspective"),
    )


def _perspective_result_payload(
    result: ValueOfPerspectiveResult,
    command: str,
) -> dict[str, object]:
    """Return a JSON/CSV-safe payload for Value of Perspective results."""
    return {
        "command": command,
        "metric": "Value of Perspective",
        "value": result.value,
        "method_maturity": result.method_maturity,
        "reporting": result.reporting,
        "perspectives": result.perspective_ids,
        "strategies": result.strategy_names,
        "optimal_strategy_by_perspective": dict(
            zip(result.perspective_ids, result.optimal_strategy_names, strict=True)
        ),
        "regret_matrix": result.regret_matrix.tolist(),
        "switching_values": result.switching_values.tolist(),
        "consensus_strategy": result.consensus_strategy_name,
        "consensus_weighted_expected_net_benefit": (
            result.consensus_weighted_expected_net_benefit
        ),
        "robust_strategy": result.robust_strategy_name,
        "pareto_strategies": result.pareto_strategy_names,
        "reference_perspective": result.reference_perspective_id,
        "diagnostics": result.diagnostics,
    }


def _scalar_result_payload(
    *,
    command: str,
    metric: str,
    value: float,
    method_family: str,
    estimator: str | None = None,
    reporting_details: dict[str, object] | None = None,
    **details: object,
) -> dict[str, object]:
    """Return a JSON/CSV-safe payload for scalar VOI results."""
    payload: dict[str, object] = {
        "command": command,
        "metric": metric,
        "value": value,
        "reporting": build_cheers_reporting(
            analysis_type=command,
            method_family=method_family,
            method_maturity="stable",
            estimator=estimator,
            reproducibility=reporting_details or {},
        ),
    }
    payload.update(details)
    return payload


def _read_plot_series(path: Path) -> dict[str, list[float]]:
    """Read a simple plot-series JSON payload."""
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise TypeError("Plot series file must contain a JSON object.")
    payload = cast("dict[str, object]", payload)
    series: dict[str, list[float]] = {}
    for key, values in payload.items():
        if isinstance(values, list):
            series[key] = [float(cast("float", value)) for value in values]
    return series


def _read_cost_effect_csv(path: Path) -> tuple[list[float], list[float], list[str]]:
    """Read dominance plot data from a CSV file with cost/effect columns."""
    costs: list[float] = []
    effects: list[float] = []
    names: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(str(row.get("strategy", f"Strategy {len(names) + 1}")))
            costs.append(float(row["cost"]))
            effects.append(float(row["effect"]))
    if not costs:
        raise ValueError("Dominance input file must contain at least one row.")
    return costs, effects, names


def _read_scalar_input(value_or_file: str, label: str) -> float:
    """Read a scalar value directly or from a result file."""
    _log_cli_debug("read-scalar", label=label, value_or_file=value_or_file)
    try:
        return float(value_or_file)
    except ValueError:
        path = Path(value_or_file)
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found at '{path}'") from None

        content = path.read_text(encoding="utf-8").strip()
        match = _SCALAR_PATTERN.search(content)
        if match is None:
            raise ValueError(
                f"{label} file does not contain a numeric value."
            ) from None
        return float(match.group(0))


def _read_float(value: object, label: str) -> float:
    """Parse a numeric JSON value."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number.")
    return float(value)


def _generate_config_template(template_name: str) -> dict[str, object]:
    """Return a named configuration template for the CLI."""
    try:
        return _CONFIG_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(sorted(_CONFIG_TEMPLATES))
        raise ValueError(
            f"Unknown config template '{template_name}'. Available templates: {available}"
        ) from exc


@app.command(name="generate-config")
def generate_config(
    template_name: str = typer.Argument(
        ...,
        help="Name of the configuration template to generate (for example: evsi)",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save the generated config"
    ),
) -> None:
    """Generate a JSON configuration template for a CLI command.

    Examples
    --------
    Generate an EVSI config template and save it to a file:

    .. code-block:: bash

        voiage generate-config evsi > evsi_config.json
    """
    try:
        _log_cli_debug(
            "generate-config",
            template_name=template_name,
            output_file=str(output_file) if output_file else None,
        )
        payload = _generate_config_template(template_name)
        output_text = json.dumps(payload, indent=2, sort_keys=True)
        typer.echo(output_text)
        if output_file:
            _write_output_file(output_file, output_text)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_evpi(
    net_benefit_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV containing net benefits (samples x strategies)",
    ),
    population: float | None = typer.Option(
        None, "--population", help="Population size for population-adjusted EVPI"
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save EVPI result"
    ),
) -> None:
    """Calculate Expected Value of Perfect Information (EVPI) from input data.

    Examples
    --------
    Calculate EVPI from a net-benefits CSV:

    .. code-block:: bash

        voiage calculate-evpi net_benefits.csv
    """
    try:
        _log_cli_debug(
            "calculate-evpi",
            input_file=str(net_benefit_file),
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )
        # Read net benefit data from CSV
        nba = read_value_array_csv(str(net_benefit_file), skip_header=True)

        # Calculate EVPI
        result = evpi(
            nba,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )

        # Format result string
        result_str = f"EVPI: {result:.6f}"
        output_text = _format_output(
            result_str,
            _scalar_result_payload(
                command="calculate-evpi",
                metric="EVPI",
                value=result,
                method_family="evpi",
                estimator="analytic",
                reporting_details={
                    "population": population,
                    "discount_rate": discount_rate,
                    "time_horizon": time_horizon,
                },
            ),
        )

        # Print result to console
        typer.echo(output_text)

        # Save to output file if specified
        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError:
        typer.echo(
            f"Error: Net benefit file not found at '{net_benefit_file}'", err=True
        )
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
        help="Path to CSV containing net benefits (samples x strategies)",
    ),
    parameter_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV for parameters of interest (samples x params)",
    ),
    population: float | None = typer.Option(
        None, "--population", help="Population size for population-adjusted EVPPI"
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save EVPPI result"
    ),
) -> None:
    """Calculate Expected Value of Partial Perfect Information (EVPPI).

    Examples
    --------
    Calculate EVPPI from net-benefit and parameter sample CSV files:

    .. code-block:: bash

        voiage calculate-evppi net_benefits.csv parameters.csv
    """
    try:
        _log_cli_debug(
            "calculate-evppi",
            net_benefit_file=str(net_benefit_file),
            parameter_file=str(parameter_file),
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )
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
            time_horizon=time_horizon,
        )

        # Format result string
        result_str = f"EVPPI: {result:.6f}"
        output_text = _format_output(
            result_str,
            _scalar_result_payload(
                command="calculate-evppi",
                metric="EVPPI",
                value=result,
                method_family="evppi",
                estimator="regression",
                reporting_details={
                    "population": population,
                    "discount_rate": discount_rate,
                    "time_horizon": time_horizon,
                },
            ),
        )

        # Print result to console
        typer.echo(output_text)

        # Save to output file if specified
        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_evsi(
    parameter_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV containing PSA parameters (samples x parameters)",
    ),
    trial_design_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON trial design with an 'arms' array",
    ),
    net_benefit_file: Path | None = typer.Option(
        None,
        "--net-benefit-file",
        "-n",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV net benefits used by the default static model",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Dotted path to a callable accepting ParameterSet and returning ValueArray",
    ),
    method: str = typer.Option(
        "two_loop",
        "--method",
        help="EVSI method: two_loop, regression, efficient, or moment_based",
    ),
    metamodel: str = typer.Option(
        "linear",
        "--metamodel",
        help="Metamodel for efficient EVSI: linear or random_forest",
    ),
    n_outer_loops: int = typer.Option(
        100,
        "--n-outer-loops",
        help="Outer simulations for two_loop/regression EVSI",
    ),
    n_inner_loops: int = typer.Option(
        1000,
        "--n-inner-loops",
        help="Inner simulations for two_loop EVSI",
    ),
    population: float | None = typer.Option(
        None, "--population", help="Population size for population-adjusted EVSI"
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save EVSI result"
    ),
) -> None:
    """Calculate Expected Value of Sample Information (EVSI).

    Examples
    --------
    Calculate EVSI from a PSA parameter CSV and a trial design JSON file:

    .. code-block:: bash

        voiage calculate-evsi parameters.csv trial_design.json
    """
    try:
        _log_cli_debug(
            "calculate-evsi",
            parameter_file=str(parameter_file),
            trial_design_file=str(trial_design_file),
            net_benefit_file=str(net_benefit_file) if net_benefit_file else None,
            model=model,
            method=method,
            metamodel=metamodel,
            n_outer_loops=n_outer_loops,
            n_inner_loops=n_inner_loops,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )
        param_set = read_parameter_set_csv(str(parameter_file), skip_header=True)
        trial_design = _read_trial_design_json(trial_design_file)

        if model is not None:
            model_func = import_callable(model)
        elif net_benefit_file is not None:
            nba = read_value_array_csv(str(net_benefit_file), skip_header=True)

            def model_func(_: ParameterSet) -> ValueArray:
                return nba

        else:
            typer.echo(
                "Error: provide either --model or --net-benefit-file for EVSI.",
                err=True,
            )
            raise typer.Exit(code=1)

        result = evsi(
            model_func=model_func,
            psa_prior=param_set,
            trial_design=trial_design,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
            method=method,
            n_outer_loops=n_outer_loops,
            n_inner_loops=n_inner_loops,
            metamodel=metamodel,
        )
        result_str = f"EVSI: {result:.6f}"
        output_text = _format_output(
            result_str,
            _scalar_result_payload(
                command="calculate-evsi",
                metric="EVSI",
                value=result,
                method_family="evsi",
                estimator=method,
                reporting_details={
                    "population": population,
                    "discount_rate": discount_rate,
                    "time_horizon": time_horizon,
                    "metamodel": metamodel,
                    "n_outer_loops": n_outer_loops,
                    "n_inner_loops": n_inner_loops,
                },
            ),
        )

        typer.echo(output_text)

        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in trial design file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_enbs(
    evsi_input: str = typer.Option(
        ...,
        "--evsi",
        help="EVSI value or path to a file containing a numeric EVSI result",
    ),
    research_cost: float = typer.Option(
        ...,
        "--research-cost",
        help="Total cost of the proposed research",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save ENBS result"
    ),
) -> None:
    """Calculate Expected Net Benefit of Sampling (ENBS).

    Examples
    --------
    Calculate ENBS from an EVSI value and research cost:

    .. code-block:: bash

        voiage calculate-enbs --evsi 12.5 --research-cost 10.0
    """
    try:
        _log_cli_debug(
            "calculate-enbs",
            evsi_input=evsi_input,
            research_cost=research_cost,
        )
        evsi_result = _read_scalar_input(evsi_input, "EVSI")
        result = enbs(evsi_result=evsi_result, research_cost=research_cost)
        result_str = f"ENBS: {result:.6f}"
        output_text = _format_output(
            result_str,
            _scalar_result_payload(
                command="calculate-enbs",
                metric="ENBS",
                value=result,
                method_family="enbs",
                estimator="difference",
                evsi_input=evsi_input,
                research_cost=research_cost,
                reporting_details={
                    "evsi_input": evsi_input,
                    "research_cost": research_cost,
                },
            ),
        )
        typer.echo(output_text)

        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_adaptive_evsi(
    parameter_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV containing PSA parameters (samples x parameters)",
    ),
    trial_design_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON trial design with arm definitions",
    ),
    adaptive_rules_file: Path = typer.Option(
        ...,
        "--adaptive-rules",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON adaptive rules specification",
    ),
    simulator: str = typer.Option(
        "bayesian",
        "--simulator",
        help="Adaptive simulator to use: bayesian or sophisticated",
    ),
    population: float | None = typer.Option(
        None, "--population", help="Population size for population-adjusted EVSI"
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    n_outer_loops: int = typer.Option(
        10, "--n-outer-loops", help="Outer Monte Carlo loops"
    ),
    n_inner_loops: int = typer.Option(
        50, "--n-inner-loops", help="Inner Monte Carlo loops"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save adaptive EVSI result"
    ),
) -> None:
    """Calculate Expected Value of Sample Information for adaptive trial designs.

    Examples
    --------
    Calculate adaptive EVSI from parameters, a trial design, and rules:

    .. code-block:: bash

        voiage calculate-adaptive-evsi parameters.csv trial_design.json --adaptive-rules adaptive_rules.json
    """
    try:
        _log_cli_debug(
            "calculate-adaptive-evsi",
            parameter_file=str(parameter_file),
            trial_design_file=str(trial_design_file),
            adaptive_rules_file=str(adaptive_rules_file),
            simulator=simulator,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
            n_outer_loops=n_outer_loops,
            n_inner_loops=n_inner_loops,
        )
        param_set = read_parameter_set_csv(str(parameter_file), skip_header=True)
        trial_design = _read_trial_design_json(trial_design_file)
        adaptive_rules_obj = _read_json_file(adaptive_rules_file)
        if not isinstance(adaptive_rules_obj, dict):
            raise TypeError("Adaptive rules file must contain a JSON object.")
        adaptive_rules = cast("dict[str, object]", adaptive_rules_obj)

        if simulator == "bayesian":
            adaptive_simulator = bayesian_adaptive_trial_simulator
        elif simulator == "sophisticated":
            adaptive_simulator = sophisticated_adaptive_trial_simulator
        else:
            raise ValueError("simulator must be one of: bayesian, sophisticated.")

        result = adaptive_evsi(
            adaptive_trial_simulator=adaptive_simulator,
            psa_prior=param_set,
            base_trial_design=trial_design,
            adaptive_rules=adaptive_rules,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
            n_outer_loops=n_outer_loops,
            n_inner_loops=n_inner_loops,
        )
        result_str = f"Adaptive EVSI: {result:.6f}"
        output_text = _format_output(
            result_str,
            {
                "command": "calculate-adaptive-evsi",
                "metric": "Adaptive EVSI",
                "value": result,
                "simulator": simulator,
            },
        )
        typer.echo(output_text)

        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in input file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def calculate_portfolio_voi(
    portfolio_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON portfolio specification",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save portfolio VOI result"
    ),
) -> None:
    """Optimize a research portfolio using the portfolio VOI methods.

    Examples
    --------
    Optimize a portfolio from a JSON specification:

    .. code-block:: bash

        voiage calculate-portfolio-voi portfolio.json
    """
    try:
        _log_cli_debug(
            "calculate-portfolio-voi",
            portfolio_file=str(portfolio_file),
        )
        config_obj = _read_json_file(portfolio_file)
        if not isinstance(config_obj, dict):
            raise TypeError("Portfolio file must contain a JSON object.")
        config = cast("dict[str, object]", config_obj)

        studies_data = config.get("studies")
        if not isinstance(studies_data, list) or not studies_data:
            raise TypeError("Portfolio file must contain a non-empty 'studies' list.")
        studies_data = cast("list[object]", studies_data)

        studies: list[PortfolioStudy] = []
        study_values: dict[str, float] = {}
        for study_data in studies_data:
            if not isinstance(study_data, dict):
                raise TypeError("Each portfolio study must be a JSON object.")
            study_data = cast("dict[str, object]", study_data)
            if "name" not in study_data or "cost" not in study_data:
                raise TypeError("Each study must include 'name' and 'cost'.")
            if "design" not in study_data or "value" not in study_data:
                raise TypeError("Each study must include 'design' and 'value'.")

            name = str(study_data["name"])
            cost = _read_float(study_data["cost"], f"Study '{name}' cost")
            value = _read_float(study_data["value"], f"Study '{name}' value")
            design_obj = study_data["design"]
            if not isinstance(design_obj, dict):
                raise TypeError(f"Study '{name}' design must be a JSON object.")

            studies.append(
                PortfolioStudy(
                    name=name,
                    design=TrialDesign.from_dict(design_obj),
                    cost=cost,
                )
            )
            study_values[name] = value

        budget_constraint = None
        if "budget_constraint" in config and config["budget_constraint"] is not None:
            budget_constraint = _read_float(
                config["budget_constraint"], "budget_constraint"
            )

        portfolio_spec = PortfolioSpec(
            studies=studies,
            budget_constraint=budget_constraint,
        )

        optimization_method = str(config.get("optimization_method", "greedy"))
        dependency_groups = config.get("dependency_groups")
        dependency_discount = _read_float(
            config.get("dependency_discount", 0.0), "dependency_discount"
        )

        result = portfolio_voi(
            portfolio_specification=portfolio_spec,
            study_value_calculator=lambda study: study_values[study.name],
            optimization_method=optimization_method,
            dependency_groups=dependency_groups
            if isinstance(dependency_groups, dict)
            else None,
            dependency_discount=dependency_discount,
        )

        selected_studies = cast("list[PortfolioStudy]", result["selected_studies"])
        selected_names = ", ".join(study.name for study in selected_studies)
        if not selected_names:
            selected_names = "None"

        total_value = float(cast("float", result["total_value"]))
        total_cost = float(cast("float", result["total_cost"]))
        text_output = "\n".join(
            [
                f"Selected studies: {selected_names}",
                f"Total value: {total_value:.6f}",
                f"Total cost: {total_cost:.6f}",
            ]
        )
        output_text = _format_output(
            text_output,
            {
                "command": "calculate-portfolio-voi",
                "selected_studies": [study.name for study in selected_studies],
                "total_value": total_value,
                "total_cost": total_cost,
                "optimization_method": optimization_method,
            },
        )
        typer.echo(output_text)

        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in portfolio file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


def _sequential_step_stub(
    psa: ParameterSet, action: object, specification: DynamicSpec
) -> dict[str, object]:
    """Minimal placeholder step model for the CLI sequential VOI wrapper."""
    _ = psa, action, specification
    return {}


@app.command()
def calculate_sequential_voi(
    parameter_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV containing PSA parameters (samples x parameters)",
    ),
    dynamic_spec_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON dynamic specification with time_steps",
    ),
    wtp: float = typer.Option(0.0, "--wtp", help="Willingness-to-pay threshold"),
    population: float | None = typer.Option(
        None, "--population", help="Population size for scaling"
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    optimization_method: str = typer.Option(
        "backward_induction",
        "--optimization-method",
        help="Sequential optimization method",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save sequential VOI result"
    ),
) -> None:
    """Calculate sequential VOI using a dynamic specification.

    Examples
    --------
    Calculate sequential VOI from a parameter CSV and dynamic spec JSON:

    .. code-block:: bash

        voiage calculate-sequential-voi parameters.csv dynamic_spec.json
    """
    try:
        _log_cli_debug(
            "calculate-sequential-voi",
            parameter_file=str(parameter_file),
            dynamic_spec_file=str(dynamic_spec_file),
            wtp=wtp,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
            optimization_method=optimization_method,
        )
        initial_psa = read_parameter_set_csv(str(parameter_file), skip_header=True)
        dynamic_spec_obj = _read_json_file(dynamic_spec_file)
        if not isinstance(dynamic_spec_obj, dict):
            raise TypeError("Dynamic spec file must contain a JSON object.")
        dynamic_spec_obj = cast("dict[str, object]", dynamic_spec_obj)
        if "time_steps" not in dynamic_spec_obj:
            raise TypeError("Dynamic spec file must contain 'time_steps'.")

        time_steps = cast("Sequence[float]", dynamic_spec_obj["time_steps"])
        dynamic_spec = DynamicSpec(time_steps=time_steps)
        result = sequential_voi(
            step_model=_sequential_step_stub,
            initial_psa=initial_psa,
            dynamic_specification=dynamic_spec,
            wtp=wtp,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
            optimization_method=optimization_method,
        )

        if not isinstance(result, (int, float)):
            raise TypeError("Sequential VOI did not return a numeric result.")

        result_str = f"Sequential VOI: {float(result):.6f}"
        output_text = _format_output(
            result_str,
            {
                "command": "calculate-sequential-voi",
                "metric": "Sequential VOI",
                "value": float(result),
                "wtp": wtp,
                "optimization_method": optimization_method,
            },
        )
        typer.echo(output_text)

        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in dynamic spec file - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
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
        help="Path to JSON config file defining model structures",
    ),
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted structural EVPI",
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save structural EVPI result"
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

    Examples
    --------
    Calculate structural EVPI from a JSON config:

    .. code-block:: bash

        voiage calculate-structural-evpi structural_config.json
    """
    try:
        _log_cli_debug(
            "calculate-structural-evpi",
            config_file=str(config_file),
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )
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
            typer.echo(
                f"Error: Structure probabilities must sum to 1 (got {total_prob})",
                err=True,
            )
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
            def make_evaluator(nb_array: object) -> Callable[[object], object]:
                def evaluator(psa_sample: object) -> object:
                    return nb_array

                return evaluator

            evaluators.append(make_evaluator(nba))

            # Create PSA sample (use net benefit values as proxy)
            psa = ParameterSet.from_numpy_or_dict(
                {
                    f"param_{i}": nba.numpy_values[:, i]
                    if i < nba.numpy_values.shape[1]
                    else nba.numpy_values[:, 0]
                    for i in range(max(1, nba.numpy_values.shape[1]))
                }
            )
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
            time_horizon=time_horizon,
        )

        # Format result string
        result_str = f"Structural EVPI: {result:.6f}"
        output_text = _format_output(
            result_str,
            {
                "command": "calculate-structural-evpi",
                "metric": "Structural EVPI",
                "value": result,
            },
        )

        # Print result to console
        typer.echo(output_text)

        # Save to output file if specified
        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
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
        help="Path to JSON config file defining model structures",
    ),
    structures_of_interest: list[int] | None = typer.Option(
        None,
        "--structures-of-interest",
        "-s",
        help="Indices of structures to learn about (0-indexed, can specify multiple)",
    ),
    parameters_of_interest: list[int] | None = typer.Option(
        None,
        "--parameters-of-interest",
        "-p",
        help="Alias for --structures-of-interest for structural EVPPI",
    ),
    population: float | None = typer.Option(
        None,
        "--population",
        help="Population size for population-adjusted structural EVPPI",
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save structural EVPPI result"
    ),
) -> None:
    """Calculate Structural Expected Value of Partial Perfect Information (SEVPPI).

    Structural EVPPI quantifies the expected gain from resolving uncertainty
    about a specific subset of model structures.

    The config file format is the same as for structural EVPI.

    Examples
    --------
    Calculate structural EVPPI for selected structures:

    .. code-block:: bash

        voiage calculate-structural-evppi structural_config.json --structures-of-interest 0 2
    """
    try:
        _log_cli_debug(
            "calculate-structural-evppi",
            config_file=str(config_file),
            structures_of_interest=structures_of_interest,
            parameters_of_interest=parameters_of_interest,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )
        selected_structures = structures_of_interest or parameters_of_interest
        if not selected_structures:
            typer.echo(
                "Error: --structures-of-interest or --parameters-of-interest is required",
                err=True,
            )
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
            typer.echo(
                f"Error: Structure probabilities must sum to 1 (got {total_prob})",
                err=True,
            )
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
            def make_evaluator(nb_array: object) -> Callable[[object], object]:
                def evaluator(psa_sample: object) -> object:
                    return nb_array

                return evaluator

            evaluators.append(make_evaluator(nba))

            # Create PSA sample
            psa = ParameterSet.from_numpy_or_dict(
                {
                    f"param_{i}": nba.numpy_values[:, i]
                    if i < nba.numpy_values.shape[1]
                    else nba.numpy_values[:, 0]
                    for i in range(max(1, nba.numpy_values.shape[1]))
                }
            )
            psa_samples.append(psa)

        # Extract probabilities
        probabilities = [s["probability"] for s in structures]

        # Calculate structural EVPPI
        result = structural_evppi(
            evaluators,
            probabilities,
            psa_samples,
            structures_of_interest=list(selected_structures),
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )

        # Format result string
        result_str = f"Structural EVPPI: {result:.6f}"
        output_text = _format_output(
            result_str,
            {
                "command": "calculate-structural-evppi",
                "metric": "Structural EVPPI",
                "value": result,
                "structures_of_interest": list(selected_structures),
            },
        )

        # Print result to console
        typer.echo(output_text)

        # Save to output file if specified
        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
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
        help="Path to JSON config file defining NMA data",
    ),
    parameters_of_interest: str | None = typer.Option(
        None,
        "--parameters-of-interest",
        "-p",
        help="Comma-separated list of parameters for EVPPI (e.g., 'effect_A,effect_B')",
    ),
    willingness_to_pay: float | None = typer.Option(
        None, "--willingness-to-pay", "-w", help="Willingness-to-pay threshold per unit"
    ),
    population: float | None = typer.Option(
        None, "--population", help="Population size for population-adjusted VOI"
    ),
    discount_rate: float | None = typer.Option(
        None, "--discount-rate", help="Annual discount rate (e.g., 0.03)"
    ),
    time_horizon: float | None = typer.Option(
        None, "--time-horizon", help="Time horizon in years"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save NMA VOI result"
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

    Examples
    --------
    Calculate NMA EVPI from a JSON config:

    .. code-block:: bash

        voiage calculate-nma-voi nma_config.json
    """
    try:
        _log_cli_debug(
            "calculate-nma-voi",
            config_file=str(config_file),
            parameters_of_interest=parameters_of_interest,
            willingness_to_pay=willingness_to_pay,
            population=population,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
        )
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
        output_text = _format_output(
            result_str,
            {
                "command": "calculate-nma-voi",
                "metric": "NMA-EVPPI" if parameters_of_interest else "NMA-EVPI",
                "value": result,
                "parameters_of_interest": params_list if parameters_of_interest else [],
            },
        )

        # Print result to console
        typer.echo(output_text)

        # Save to output file if specified
        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
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
def calculate_perspective(
    surface_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON surface data for Value of Perspective",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save Value of Perspective result"
    ),
) -> None:
    """Calculate experimental Value of Perspective from a JSON surface file.

    Examples
    --------
    Calculate perspective regret and consensus strategy:

    .. code-block:: bash

        voiage calculate-perspective perspective_surface.json
    """
    try:
        _log_cli_debug(
            "calculate-perspective",
            surface_file=str(surface_file),
            output_file=str(output_file) if output_file else None,
        )
        (
            value_array,
            strategy_names,
            perspective_names,
            perspective_weights,
            reference_perspective,
        ) = _read_perspective_surface(surface_file)
        result = calculate_perspective_result(
            value_array,
            strategy_names=strategy_names,
            perspective_names=perspective_names,
            perspective_weights=cast(
                "Sequence[float] | dict[str, float] | None",
                perspective_weights,
            ),
            reference_perspective=cast("str | int | None", reference_perspective),
        )
        result_str = (
            f"Value of Perspective: {result.value:.6f}\n"
            f"Consensus strategy: {result.consensus_strategy_name}\n"
            f"Robust strategy: {result.robust_strategy_name}"
        )
        output_text = _format_output(
            result_str,
            _perspective_result_payload(result, "calculate-perspective"),
        )

        typer.echo(output_text)

        if output_file:
            _write_output_file(output_file, output_text)
            if _should_echo_status_messages():
                typer.echo(f"Result saved to {output_file}")

    except FileNotFoundError:
        typer.echo(f"Error: Surface file not found at '{surface_file}'", err=True)
        raise typer.Exit(code=1) from None
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def plot_perspective_regret(
    surface_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON surface data for a Value of Perspective regret plot",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save the perspective regret plot"
    ),
) -> None:
    """Plot the experimental Value of Perspective regret matrix.

    Examples
    --------
    Plot a perspective regret matrix:

    .. code-block:: bash

        voiage plot-perspective-regret perspective_surface.json --output regret.png
    """
    try:
        _log_cli_debug(
            "plot-perspective-regret",
            surface_file=str(surface_file),
            output_file=str(output_file) if output_file else None,
        )
        (
            value_array,
            strategy_names,
            perspective_names,
            perspective_weights,
            reference_perspective,
        ) = _read_perspective_surface(surface_file)
        result = calculate_perspective_result(
            value_array,
            strategy_names=strategy_names,
            perspective_names=perspective_names,
            perspective_weights=cast(
                "Sequence[float] | dict[str, float] | None",
                perspective_weights,
            ),
            reference_perspective=cast("str | int | None", reference_perspective),
        )
        ax = render_perspective_regret(result)
        if _should_echo_status_messages():
            if output_file:
                _save_figure(ax, output_file)
                typer.echo(f"Plot saved to {output_file}")
        else:
            if output_file:
                _save_figure(ax, output_file)
            typer.echo(
                _format_output(
                    "Plot generated",
                    {
                        "command": "plot-perspective-regret",
                        "input_file": str(surface_file),
                        "output_file": str(output_file) if output_file else None,
                        "saved": output_file is not None,
                    },
                )
            )
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def plot_ceac(
    surface_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON surface data with net_benefit and wtp_thresholds",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save the CEAC plot"
    ),
) -> None:
    """Plot a Cost-Effectiveness Acceptability Curve.

    Examples
    --------
    Plot a CEAC from a JSON surface file:

    .. code-block:: bash

        voiage plot-ceac surface.json
    """
    try:
        _log_cli_debug(
            "plot-ceac",
            surface_file=str(surface_file),
            output_file=str(output_file) if output_file else None,
        )
        value_array, wtp_thresholds = _read_plot_surface(surface_file)
        ax = render_ceac(value_array, wtp_thresholds=wtp_thresholds)
        if _should_echo_status_messages():
            if output_file:
                _save_figure(ax, output_file)
                typer.echo(f"Plot saved to {output_file}")
        else:
            if output_file:
                _save_figure(ax, output_file)
            typer.echo(
                _format_output(
                    "Plot generated",
                    {
                        "command": "plot-ceac",
                        "input_file": str(surface_file),
                        "output_file": str(output_file) if output_file else None,
                        "saved": output_file is not None,
                    },
                )
            )
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def plot_ceaf(
    surface_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON surface data with net_benefit and wtp_thresholds",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save the CEAF plot"
    ),
) -> None:
    """Plot a Cost-Effectiveness Acceptability Frontier.

    Examples
    --------
    Plot a CEAF from a JSON surface file:

    .. code-block:: bash

        voiage plot-ceaf surface.json
    """
    try:
        _log_cli_debug(
            "plot-ceaf",
            surface_file=str(surface_file),
            output_file=str(output_file) if output_file else None,
        )
        value_array, wtp_thresholds = _read_plot_surface(surface_file)
        result = calculate_ceaf_result(value_array, wtp_thresholds)
        ax = render_ceaf(value_array, wtp_thresholds=wtp_thresholds, result=result)
        if _should_echo_status_messages():
            if output_file:
                _save_figure(ax, output_file)
                typer.echo(f"Plot saved to {output_file}")
        else:
            if output_file:
                _save_figure(ax, output_file)
            typer.echo(
                _format_output(
                    "Plot generated",
                    {
                        "command": "plot-ceaf",
                        "input_file": str(surface_file),
                        "output_file": str(output_file) if output_file else None,
                        "saved": output_file is not None,
                    },
                )
            )
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def plot_voi_curves(
    series_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSON plot series data",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save the VOI curves plot"
    ),
) -> None:
    """Plot EVPI/EVSI VOI curves from JSON series data.

    Examples
    --------
    Plot VOI curves from a JSON series file:

    .. code-block:: bash

        voiage plot-voi-curves curves.json
    """
    try:
        _log_cli_debug(
            "plot-voi-curves",
            series_file=str(series_file),
            output_file=str(output_file) if output_file else None,
        )
        payload = _read_plot_series(series_file)
        if "evpi_values" in payload and "wtp_thresholds" in payload:
            ax = render_evpi_vs_wtp(
                evpi_values=payload["evpi_values"],
                wtp_thresholds=payload["wtp_thresholds"],
            )
        elif "evsi_values" in payload and "sample_sizes" in payload:
            ax = render_evsi_vs_sample_size(
                evsi_values=payload["evsi_values"],
                sample_sizes=payload["sample_sizes"],
                enbs_values=payload.get("enbs_values"),
                research_costs=payload.get("research_costs"),
            )
        else:
            raise TypeError("Plot series file must contain either EVPI or EVSI data.")

        if _should_echo_status_messages():
            if output_file:
                _save_figure(ax, output_file)
                typer.echo(f"Plot saved to {output_file}")
        else:
            if output_file:
                _save_figure(ax, output_file)
            typer.echo(
                _format_output(
                    "Plot generated",
                    {
                        "command": "plot-voi-curves",
                        "input_file": str(series_file),
                        "output_file": str(output_file) if output_file else None,
                        "saved": output_file is not None,
                    },
                )
            )
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def plot_dominance(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to CSV file with strategy,cost,effect columns",
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="File to save the dominance plot"
    ),
) -> None:
    """Plot the cost-effectiveness plane and frontier.

    Examples
    --------
    Plot dominance from a CSV file:

    .. code-block:: bash

        voiage plot-dominance dominance.csv
    """
    try:
        _log_cli_debug(
            "plot-dominance",
            input_file=str(input_file),
            output_file=str(output_file) if output_file else None,
        )
        costs, effects, names = _read_cost_effect_csv(input_file)
        result = calculate_dominance_result(costs, effects, strategy_names=names)
        ax = render_dominance(
            costs=costs, effects=effects, strategy_names=names, result=result
        )
        if _should_echo_status_messages():
            if output_file:
                _save_figure(ax, output_file)
                typer.echo(f"Plot saved to {output_file}")
        else:
            if output_file:
                _save_figure(ax, output_file)
            typer.echo(
                _format_output(
                    "Plot generated",
                    {
                        "command": "plot-dominance",
                        "input_file": str(input_file),
                        "output_file": str(output_file) if output_file else None,
                        "saved": output_file is not None,
                    },
                )
            )
    except FileNotFoundError as e:
        typer.echo(f"Error: File not found - {e}", err=True)
        raise typer.Exit(code=1) from e
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"An error occurred: {e}", err=True)
        raise typer.Exit(code=1) from e


# Example of how to run if using Typer directly (not through setup.py entry points)
if __name__ == "__main__":  # pragma: no cover
    app()
