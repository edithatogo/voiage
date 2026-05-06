"""Focused CLI tests for heterogeneity, distributional, and implementation outputs."""

from collections.abc import Callable
import csv
import json
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from voiage import cli

runner = CliRunner()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload))
        writer.writeheader()
        for row_values in zip(*payload.values(), strict=True):
            writer.writerow(dict(zip(payload.keys(), row_values, strict=True)))


def _json_output(result: object) -> dict[str, object]:
    return json.loads(cast("str", result.stdout))


def _csv_rows(result: object) -> list[dict[str, str]]:
    return list(csv.DictReader(cast("str", result.stdout).splitlines()))


@pytest.mark.parametrize(
    (
        "command",
        "input_name",
        "payload",
        "metric",
        "text_snippet",
        "payload_checks",
        "csv_checks",
    ),
    [
        (
            "calculate-heterogeneity",
            "heterogeneity.json",
            {
                "net_benefit": [
                    [9.0, 7.0],
                    [10.0, 8.0],
                    [5.0, 11.0],
                    [6.0, 12.0],
                ],
                "strategy_names": ["A", "B"],
                "subgroups": ["young", "young", "old", "old"],
            },
            "Value of Heterogeneity",
            "Value of Heterogeneity: 1.000000",
            lambda payload: (
                payload["method_maturity"] == "stable"
                and payload["subgroup_labels"] == ["old", "young"]
                and payload["subgroup_optimal_strategy_names"] == ["B", "A"]
                and payload["overall_optimal_strategy_name"] == "B"
                and payload["reporting"]["analysis_type"] == "value_of_heterogeneity"
            ),
            lambda row: (
                json.loads(row["reporting"])["method_maturity"] == "stable"
                and json.loads(row["subgroup_labels"]) == ["old", "young"]
                and json.loads(row["subgroup_expected_net_benefits"]) == [11.5, 9.5]
            ),
        ),
        (
            "calculate-ceaf",
            "ceaf.json",
            {
                "strategy_names": ["A", "B"],
                "wtp_thresholds": [0.0, 50000.0],
                "net_benefit": [
                    [[10.0, 15.0], [12.0, 14.0]],
                    [[11.0, 16.0], [13.0, 15.0]],
                ],
            },
            "CEAF",
            "CEAF computed across 2 thresholds",
            lambda payload: (
                payload["reporting"]["analysis_type"] == "calculate_ceaf"
                and payload["optimal_strategy_names"] == ["B", "A"]
                and payload["wtp_thresholds"] == [0.0, 50000.0]
            ),
            lambda row: (
                json.loads(row["reporting"])["analysis_type"] == "calculate_ceaf"
                and json.loads(row["optimal_strategy_names"]) == ["B", "A"]
                and json.loads(row["wtp_thresholds"]) == [0.0, 50000.0]
            ),
        ),
        (
            "calculate-dominance",
            "dominance.csv",
            {
                "strategy": ["A", "B", "C"],
                "cost": [100.0, 102.0, 104.0],
                "effect": [1.0, 1.1, 1.2],
                },
                "Dominance",
                "Dominance frontier strategies: 3",
                lambda payload: (
                    payload["reporting"]["analysis_type"] == "calculate_dominance"
                    and payload["frontier_indices"] == [0, 1, 2]
                    and payload["strategy_names"] == ["A", "B", "C"]
                ),
                lambda row: (
                    json.loads(row["reporting"])["analysis_type"]
                    == "calculate_dominance"
                    and json.loads(row["frontier_indices"]) == [0, 1, 2]
                    and json.loads(row["strategy_names"]) == ["A", "B", "C"]
                ),
            ),
        (
            "calculate-distributional-equity",
            "distributional.json",
            {
                "net_benefit": [
                    [9.0, 7.0],
                    [10.0, 8.0],
                    [5.0, 11.0],
                    [6.0, 12.0],
                ],
                "strategy_names": ["A", "B"],
                "subgroups": ["young", "young", "old", "old"],
                "equity_weights": {"young": 2.0, "old": 1.0},
            },
            "Value of Distributional Equity",
            "Value of Distributional Equity: 1.000000",
            lambda payload: (
                payload["method_maturity"] == "experimental"
                and payload["subgroup_labels"] == ["old", "young"]
                and payload["equity_weights"] == pytest.approx([1 / 3, 2 / 3])
                and payload["social_welfare_optimal_strategy_name"] == "B"
                and payload["reporting"]["analysis_type"]
                == "value_of_distributional_equity"
            ),
            lambda row: (
                json.loads(row["reporting"])["method_maturity"] == "experimental"
                and json.loads(row["equity_weights"]) == pytest.approx([1 / 3, 2 / 3])
                and json.loads(row["equity_weighted_expected_net_benefits"])
                == pytest.approx([8.166666666666666, 8.833333333333334])
            ),
        ),
        (
            "calculate-implementation",
            "implementation.json",
            {
                "net_benefit": [
                    [10.0, 8.0],
                    [10.0, 8.0],
                    [10.0, 8.0],
                ],
                "strategy_names": ["A", "B"],
                "uptake": 0.5,
                "adherence": 1.0,
                "coverage": 1.0,
                "implementation_delay": 0.0,
                "implementation_uncertainty": 0.0,
                "discount_rate": 0.0,
            },
            "Value of Implementation",
            "Value of Implementation: 5.000000",
            lambda payload: (
                payload["method_maturity"] == "experimental"
                and payload["baseline_optimal_strategy_name"] == "A"
                and payload["adjusted_optimal_strategy_name"] == "A"
                and payload["implementation_multiplier"] == pytest.approx(0.5)
                and payload["reporting"]["analysis_type"] == "value_of_implementation"
            ),
            lambda row: (
                json.loads(row["reporting"])["method_maturity"] == "experimental"
                and json.loads(row["adjusted_expected_net_benefits"]) == [5.0, 4.0]
                and float(row["implementation_multiplier"]) == pytest.approx(0.5)
            ),
        ),
    ],
)
def test_frontier_cli_commands(
    tmp_path: Path,
    command: str,
    input_name: str,
    payload: dict[str, object],
    metric: str,
    text_snippet: str,
    payload_checks: Callable[[dict[str, object]], bool],
    csv_checks: Callable[[dict[str, str]], bool],
) -> None:
    """Exercise the new frontier CLI commands across output formats."""
    input_file = tmp_path / input_name
    if command == "calculate-dominance":
        _write_csv(input_file, payload)
    else:
        _write_json(input_file, payload)

    text_result = runner.invoke(cli.app, [command, str(input_file)])
    assert text_result.exit_code == 0
    assert text_snippet in text_result.stdout

    json_result = runner.invoke(
        cli.app,
        ["--format", "json", command, str(input_file)],
    )
    assert json_result.exit_code == 0
    json_payload = _json_output(json_result)
    assert json_payload["command"] == command
    assert json_payload["metric"] == metric
    assert json_payload["reporting"]["reporting_standard"] == "CHEERS-VOI"
    assert payload_checks(json_payload)

    csv_result = runner.invoke(
        cli.app,
        ["--format", "csv", command, str(input_file)],
    )
    assert csv_result.exit_code == 0
    csv_rows = _csv_rows(csv_result)
    assert len(csv_rows) == 1
    assert csv_rows[0]["command"] == command
    assert csv_rows[0]["metric"] == metric
    assert csv_checks(csv_rows[0])


@pytest.mark.parametrize(
    ("command", "input_name", "payload"),
    [
        (
            "calculate-ceaf",
            "ceaf.json",
            {
                "strategy_names": ["A", "B"],
                "wtp_thresholds": [0.0, 50000.0],
                "net_benefit": [
                    [[10.0, 15.0], [12.0, 14.0]],
                    [[11.0, 16.0], [13.0, 15.0]],
                ],
            },
        ),
        (
            "calculate-dominance",
            "dominance.csv",
            {
                "strategy": ["A", "B", "C"],
                "cost": [100.0, 102.0, 104.0],
                "effect": [1.0, 1.1, 1.2],
            },
        ),
        (
            "calculate-heterogeneity",
            "heterogeneity.json",
            {
                "net_benefit": [
                    [9.0, 7.0],
                    [10.0, 8.0],
                    [5.0, 11.0],
                    [6.0, 12.0],
                ],
                "strategy_names": ["A", "B"],
                "subgroups": ["young", "young", "old", "old"],
            },
        ),
        (
            "calculate-distributional-equity",
            "distributional.json",
            {
                "net_benefit": [
                    [9.0, 7.0],
                    [10.0, 8.0],
                    [5.0, 11.0],
                    [6.0, 12.0],
                ],
                "strategy_names": ["A", "B"],
                "subgroups": ["young", "young", "old", "old"],
                "equity_weights": {"young": 2.0, "old": 1.0},
            },
        ),
        (
            "calculate-implementation",
            "implementation.json",
            {
                "net_benefit": [
                    [10.0, 8.0],
                    [10.0, 8.0],
                    [10.0, 8.0],
                ],
                "strategy_names": ["A", "B"],
                "uptake": 0.5,
                "adherence": 1.0,
                "coverage": 1.0,
                "implementation_delay": 0.0,
                "implementation_uncertainty": 0.0,
                "discount_rate": 0.0,
            },
        ),
    ],
)
def test_frontier_cli_commands_write_output_file(
    tmp_path: Path,
    command: str,
    input_name: str,
    payload: dict[str, object],
) -> None:
    """Exercise the file-output branch for the frontier CLI commands."""
    input_file = tmp_path / input_name
    output_file = tmp_path / f"{input_name}.out"
    if command == "calculate-dominance":
        _write_csv(input_file, cast("dict[str, list[object]]", payload))
    else:
        _write_json(input_file, payload)

    result = runner.invoke(
        cli.app,
        [command, str(input_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Result saved to" in result.stdout


@pytest.mark.parametrize(
    ("command", "input_name", "payload", "error_text"),
    [
        (
            "calculate-ceaf",
            "ceaf_invalid.json",
            {"strategy_names": ["A", "B"], "net_benefit": [[[1.0, 2.0], [3.0, 4.0]]]},
            "wtp_thresholds",
        ),
        (
            "calculate-dominance",
            "dominance_invalid.csv",
            {"strategy": ["A"], "cost": [100.0]},
            "effect",
        ),
        (
            "calculate-heterogeneity",
            "heterogeneity_invalid.json",
            {
                "net_benefit": [[9.0, 7.0], [10.0, 8.0]],
                "strategy_names": ["A", "B"],
            },
            "must contain 'subgroups'",
        ),
        (
            "calculate-distributional-equity",
            "distributional_invalid.json",
            {
                "net_benefit": [[9.0, 7.0], [10.0, 8.0]],
                "strategy_names": ["A", "B"],
                "subgroups": ["young", "old"],
                "equity_weights": 1.0,
            },
            "'equity_weights' must be a list or mapping",
        ),
        (
            "calculate-implementation",
            "implementation_invalid.json",
            {
                "net_benefit": [[10.0, 8.0], [10.0, 8.0]],
                "strategy_names": ["A", "B"],
                "uptake": "bad",
            },
            "must be a number",
        ),
    ],
)
def test_frontier_cli_commands_reject_invalid_inputs(
    tmp_path: Path,
    command: str,
    input_name: str,
    payload: dict[str, object],
    error_text: str,
) -> None:
    """Exercise the validation branches for the new frontier CLI commands."""
    input_file = tmp_path / input_name
    if command == "calculate-dominance":
        _write_csv(input_file, cast("dict[str, list[object]]", payload))
    else:
        _write_json(input_file, payload)

    result = runner.invoke(cli.app, [command, str(input_file)])
    assert result.exit_code != 0
    assert error_text in result.stdout or error_text in result.stderr
