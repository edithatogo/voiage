"""Focused CLI coverage for adaptive EVSI and portfolio VOI branches."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from voiage import cli
from voiage.schema import PortfolioStudy, TrialDesign

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _adaptive_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    parameter_file = tmp_path / "parameters.csv"
    _write_csv(parameter_file, [["effectiveness"], [0.5], [0.6]])

    trial_design_file = tmp_path / "trial_design.json"
    _write_json(
        trial_design_file,
        {
            "arms": [
                {"name": "Control", "sample_size": 10},
                {"name": "Treatment", "sample_size": 10},
            ]
        },
    )

    adaptive_rules_file = tmp_path / "adaptive_rules.json"
    _write_json(
        adaptive_rules_file,
        {"interim_analysis_points": [0.5], "early_stopping_rules": {"efficacy": 0.9}},
    )

    return parameter_file, trial_design_file, adaptive_rules_file


def _portfolio_payload(studies: object) -> dict[str, object]:
    return {
        "budget_constraint": 120,
        "optimization_method": "dynamic_programming",
        "studies": studies,
    }


def test_calculate_adaptive_evsi_rejects_invalid_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid simulator names should fail before the adaptive VOI call runs."""
    parameter_file, trial_design_file, adaptive_rules_file = _adaptive_inputs(tmp_path)

    monkeypatch.setattr(
        cli,
        "adaptive_evsi",
        lambda **kwargs: pytest.fail(f"adaptive_evsi should not run: {kwargs!r}"),
    )

    result = runner.invoke(
        cli.app,
        [
            "calculate-adaptive-evsi",
            str(parameter_file),
            str(trial_design_file),
            "--adaptive-rules",
            str(adaptive_rules_file),
            "--simulator",
            "invalid",
        ],
    )

    assert result.exit_code == 1
    assert "simulator must be one of" in result.stderr


def test_calculate_adaptive_evsi_rejects_non_object_adaptive_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adaptive rules payloads should be validated before execution."""
    parameter_file, trial_design_file, adaptive_rules_file = _adaptive_inputs(tmp_path)
    _write_json(adaptive_rules_file, ["bad", "rules"])

    monkeypatch.setattr(
        cli,
        "adaptive_evsi",
        lambda **kwargs: pytest.fail(f"adaptive_evsi should not run: {kwargs!r}"),
    )

    result = runner.invoke(
        cli.app,
        [
            "calculate-adaptive-evsi",
            str(parameter_file),
            str(trial_design_file),
            "--adaptive-rules",
            str(adaptive_rules_file),
        ],
    )

    assert result.exit_code == 1
    assert "Adaptive rules file must contain a JSON object." in result.stderr


@pytest.mark.parametrize(
    ("simulator", "expected_simulator_name"),
    [
        ("bayesian", "bayesian_adaptive_trial_simulator"),
        ("sophisticated", "sophisticated_adaptive_trial_simulator"),
    ],
)
def test_calculate_adaptive_evsi_writes_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulator: str,
    expected_simulator_name: str,
) -> None:
    """Successful adaptive EVSI calls should take the output-file branch."""
    parameter_file, trial_design_file, adaptive_rules_file = _adaptive_inputs(tmp_path)

    seen: dict[str, object] = {}

    def fake_adaptive_evsi(**kwargs: object) -> float:
        seen.update(kwargs)
        return 2.75

    monkeypatch.setattr(cli, "adaptive_evsi", fake_adaptive_evsi)

    output_file = tmp_path / f"adaptive_evsi_{simulator}.txt"
    result = runner.invoke(
        cli.app,
        [
            "calculate-adaptive-evsi",
            str(parameter_file),
            str(trial_design_file),
            "--adaptive-rules",
            str(adaptive_rules_file),
            "--simulator",
            simulator,
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "Adaptive EVSI: 2.750000" in result.stdout
    assert f"Result saved to {output_file}" in result.stdout
    assert output_file.read_text(encoding="utf-8").strip() == "Adaptive EVSI: 2.750000"
    assert seen["adaptive_trial_simulator"].__name__ == expected_simulator_name


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (["not", "a", "dict"], "Portfolio file must contain a JSON object."),
        (_portfolio_payload([]), "non-empty 'studies' list"),
        ({}, "non-empty 'studies' list"),
        ({"studies": "not-a-list"}, "non-empty 'studies' list"),
        (_portfolio_payload([1]), "Each portfolio study must be a JSON object."),
        (
            _portfolio_payload([{"name": "Study A", "cost": 10, "design": {}}]),
            "Each study must include 'design' and 'value'.",
        ),
        (
            _portfolio_payload([{"name": "Study A", "value": 10, "design": {}}]),
            "Each study must include 'name' and 'cost'.",
        ),
        (
            _portfolio_payload(
                [{"name": "Study A", "cost": "bad", "value": 10, "design": {}}]
            ),
            "Study 'Study A' cost",
        ),
        (
            _portfolio_payload(
                [
                    {
                        "name": "Study A",
                        "cost": 10,
                        "value": 10,
                        "design": [],
                    }
                ]
            ),
            "Study 'Study A' design must be a JSON object.",
        ),
    ],
)
def test_calculate_portfolio_voi_rejects_invalid_study_payloads(
    tmp_path: Path,
    payload: object,
    expected_message: str,
) -> None:
    """Malformed portfolio study payloads should fail with a clear error."""
    portfolio_file = tmp_path / "portfolio.json"
    _write_json(portfolio_file, payload)

    result = runner.invoke(
        cli.app,
        ["calculate-portfolio-voi", str(portfolio_file)],
    )

    assert result.exit_code == 1
    assert expected_message in result.stderr


def test_calculate_portfolio_voi_writes_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful portfolio VOI calls should take the output-file branch."""
    portfolio_file = tmp_path / "portfolio.json"
    _write_json(
        portfolio_file,
        _portfolio_payload(
            [
                {
                    "name": "Study Alpha",
                    "cost": 50,
                    "value": 100,
                    "design": {"arms": [{"name": "Arm A", "sample_size": 10}]},
                },
                {
                    "name": "Study Beta",
                    "cost": 60,
                    "value": 90,
                    "design": {"arms": [{"name": "Arm B", "sample_size": 12}]},
                },
            ]
        ),
    )

    study = PortfolioStudy(
        name="Study Alpha",
        design=TrialDesign.from_dict({"arms": [{"name": "Arm A", "sample_size": 10}]}),
        cost=50.0,
    )

    def fake_portfolio_voi(**kwargs: object) -> dict[str, object]:
        assert kwargs["optimization_method"] == "dynamic_programming"
        return {
            "selected_studies": [study],
            "total_value": 100.0,
            "total_cost": 50.0,
        }

    monkeypatch.setattr(cli, "portfolio_voi", fake_portfolio_voi)

    output_file = tmp_path / "portfolio.txt"
    result = runner.invoke(
        cli.app,
        ["calculate-portfolio-voi", str(portfolio_file), "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert "Selected studies: Study Alpha" in result.stdout
    assert f"Result saved to {output_file}" in result.stdout
    assert "Selected studies: Study Alpha" in output_file.read_text(encoding="utf-8")
