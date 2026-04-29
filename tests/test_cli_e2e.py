"""End-to-end CLI smoke tests for voiage."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voiage import cli
from voiage.schema import PortfolioStudy, TrialDesign

runner = CliRunner()


class _DummyFigure:
    def savefig(self, output_file: Path, bbox_inches: str = "tight") -> None:
        _ = bbox_inches
        Path(output_file).write_text("figure", encoding="utf-8")


class _DummyAxes:
    def __init__(self) -> None:
        self.figure = _DummyFigure()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _patch_plot_renderer(monkeypatch: pytest.MonkeyPatch, renderer_name: str) -> None:
    monkeypatch.setattr(cli, renderer_name, lambda *args, **kwargs: _DummyAxes())


@pytest.mark.parametrize(
    ("command_name", "setup"),
    [
        ("calculate-evpi", "evpi"),
        ("calculate-evppi", "evppi"),
        ("calculate-evsi", "evsi"),
        ("calculate-enbs", "enbs"),
        ("calculate-adaptive-evsi", "adaptive"),
        ("calculate-portfolio-voi", "portfolio"),
        ("calculate-sequential-voi", "sequential"),
        ("calculate-structural-evpi", "structural-evpi"),
        ("calculate-structural-evppi", "structural-evppi"),
        ("calculate-nma-voi", "nma-evpi"),
        ("calculate-nma-voi", "nma-evppi"),
    ],
)
def test_cli_result_commands_e2e(
    command_name: str, setup: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke-test the result commands through the Typer CLI."""
    args: list[str]

    if setup == "evpi":
        net_benefits = tmp_path / "net_benefits.csv"
        _write_csv(
            net_benefits,
            [
                ["Strategy A", "Strategy B"],
                [1000, 1200],
                [950, 1250],
            ],
        )
        output_file = tmp_path / "evpi.json"
        args = [
            "--format",
            "json",
            command_name,
            str(net_benefits),
            "--output",
            str(output_file),
        ]
    elif setup == "evppi":
        net_benefits = tmp_path / "net_benefits.csv"
        _write_csv(
            net_benefits,
            [
                ["Strategy A", "Strategy B"],
                [1000, 1200],
                [950, 1250],
            ],
        )
        parameters = tmp_path / "parameters.csv"
        _write_csv(
            parameters,
            [
                ["param1", "param2"],
                [0.5, 0.3],
                [0.6, 0.4],
            ],
        )
        output_file = tmp_path / "evppi.json"
        args = [
            "--format",
            "json",
            command_name,
            str(net_benefits),
            str(parameters),
            "--output",
            str(output_file),
        ]
    elif setup == "evsi":
        parameters = tmp_path / "parameters.csv"
        _write_csv(parameters, [["param1"], [0.5], [0.6]])
        trial_design = tmp_path / "trial_design.json"
        _write_json(trial_design, {"arms": [{"name": "Arm A", "sample_size": 10}]})
        net_benefits = tmp_path / "net_benefits.csv"
        _write_csv(
            net_benefits,
            [
                ["Strategy A", "Strategy B"],
                [1000, 1200],
                [950, 1250],
            ],
        )
        monkeypatch.setattr(cli, "evsi", lambda **kwargs: 1.25)
        output_file = tmp_path / "evsi.json"
        args = [
            "--format",
            "json",
            command_name,
            str(parameters),
            str(trial_design),
            "--net-benefit-file",
            str(net_benefits),
            "--output",
            str(output_file),
        ]
    elif setup == "enbs":
        evsi_value = tmp_path / "evsi.txt"
        evsi_value.write_text("EVSI: 12.5\n", encoding="utf-8")
        output_file = tmp_path / "enbs.json"
        args = [
            "--format",
            "json",
            command_name,
            "--evsi",
            str(evsi_value),
            "--research-cost",
            "10.0",
            "--output",
            str(output_file),
        ]
    elif setup == "adaptive":
        parameters = tmp_path / "parameters.csv"
        _write_csv(parameters, [["param1"], [0.5], [0.6]])
        trial_design = tmp_path / "trial_design.json"
        _write_json(trial_design, {"arms": [{"name": "Arm A", "sample_size": 10}]})
        adaptive_rules = tmp_path / "adaptive_rules.json"
        _write_json(adaptive_rules, {"interim_analyses": [1]})
        monkeypatch.setattr(cli, "adaptive_evsi", lambda **kwargs: 2.5)
        output_file = tmp_path / "adaptive.json"
        args = [
            "--format",
            "json",
            command_name,
            str(parameters),
            str(trial_design),
            "--adaptive-rules",
            str(adaptive_rules),
            "--output",
            str(output_file),
        ]
    elif setup == "portfolio":
        portfolio_file = tmp_path / "portfolio.json"
        _write_json(
            portfolio_file,
            {
                "budget_constraint": 120,
                "optimization_method": "dynamic_programming",
                "studies": [
                    {
                        "name": "Study Alpha",
                        "cost": 50,
                        "value": 100,
                        "design": {"arms": [{"name": "Arm A", "sample_size": 10}]},
                    }
                ],
            },
        )
        study = PortfolioStudy(
            name="Study Alpha",
            design=TrialDesign.from_dict(
                {"arms": [{"name": "Arm A", "sample_size": 10}]}
            ),
            cost=50.0,
        )
        monkeypatch.setattr(
            cli,
            "portfolio_voi",
            lambda **kwargs: {
                "selected_studies": [study],
                "total_value": 100.0,
                "total_cost": 50.0,
            },
        )
        output_file = tmp_path / "portfolio_result.json"
        args = [
            "--format",
            "json",
            command_name,
            str(portfolio_file),
            "--output",
            str(output_file),
        ]
    elif setup == "sequential":
        parameters = tmp_path / "parameters.csv"
        _write_csv(parameters, [["param1"], [0.5], [0.6]])
        dynamic_spec = tmp_path / "dynamic_spec.json"
        _write_json(dynamic_spec, {"time_steps": [0, 1, 2]})
        monkeypatch.setattr(cli, "sequential_voi", lambda **kwargs: 3.5)
        output_file = tmp_path / "sequential.json"
        args = [
            "--format",
            "json",
            command_name,
            str(parameters),
            str(dynamic_spec),
            "--output",
            str(output_file),
        ]
    elif setup == "structural-evpi":
        config_file = tmp_path / "structural_evpi.json"
        net_benefits = tmp_path / "net_benefits.csv"
        _write_csv(
            net_benefits,
            [
                ["Strategy A", "Strategy B"],
                [1000, 1200],
                [950, 1250],
            ],
        )
        _write_json(
            config_file,
            {
                "structures": [
                    {
                        "name": "A",
                        "probability": 0.5,
                        "net_benefits_file": "net_benefits.csv",
                    },
                    {
                        "name": "B",
                        "probability": 0.5,
                        "net_benefits_file": "net_benefits.csv",
                    },
                ]
            },
        )
        monkeypatch.setattr(cli, "structural_evpi", lambda *args, **kwargs: 4.5)
        output_file = tmp_path / "structural_evpi.json"
        args = [
            "--format",
            "json",
            command_name,
            str(config_file),
            "--output",
            str(output_file),
        ]
    elif setup == "structural-evppi":
        config_file = tmp_path / "structural_evppi.json"
        net_benefits = tmp_path / "net_benefits.csv"
        _write_csv(
            net_benefits,
            [
                ["Strategy A", "Strategy B"],
                [1000, 1200],
                [950, 1250],
            ],
        )
        _write_json(
            config_file,
            {
                "structures": [
                    {
                        "name": "A",
                        "probability": 0.5,
                        "net_benefits_file": "net_benefits.csv",
                    },
                    {
                        "name": "B",
                        "probability": 0.5,
                        "net_benefits_file": "net_benefits.csv",
                    },
                ]
            },
        )
        monkeypatch.setattr(cli, "structural_evppi", lambda *args, **kwargs: 5.5)
        output_file = tmp_path / "structural_evppi.json"
        args = [
            "--format",
            "json",
            command_name,
            str(config_file),
            "--structures-of-interest",
            "0",
            "--structures-of-interest",
            "1",
            "--output",
            str(output_file),
        ]
    elif setup == "nma-evpi":
        config_file = tmp_path / "nma.json"
        _write_json(
            config_file,
            {
                "treatment_effects": {
                    "Placebo-Drug_A": [0.5, 0.6, 0.4],
                    "Placebo-Drug_B": [0.7, 0.8, 0.6],
                },
                "n_studies": 3,
                "treatments": ["Placebo", "Drug_A", "Drug_B"],
                "outcome_type": "continuous",
            },
        )
        monkeypatch.setattr(cli, "calculate_nma_evpi", lambda *args, **kwargs: 6.5)
        output_file = tmp_path / "nma.json"
        args = [
            "--format",
            "json",
            command_name,
            str(config_file),
            "--output",
            str(output_file),
        ]
    elif setup == "nma-evppi":
        config_file = tmp_path / "nma.json"
        _write_json(
            config_file,
            {
                "treatment_effects": {
                    "Placebo-Drug_A": [0.5, 0.6, 0.4],
                    "Placebo-Drug_B": [0.7, 0.8, 0.6],
                },
                "n_studies": 3,
                "treatments": ["Placebo", "Drug_A", "Drug_B"],
                "outcome_type": "continuous",
            },
        )
        monkeypatch.setattr(cli, "calculate_nma_evppi", lambda *args, **kwargs: 7.5)
        output_file = tmp_path / "nma_evppi.json"
        args = [
            "--format",
            "json",
            command_name,
            str(config_file),
            "--parameters-of-interest",
            "effect_A,effect_B",
            "--output",
            str(output_file),
        ]
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unhandled setup: {setup}")

    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == command_name
    assert Path(output_file).exists()


@pytest.mark.parametrize(
    ("command_name", "setup", "expected_output"),
    [
        ("plot-ceac", "ceac", "Plot generated"),
        ("plot-ceaf", "ceaf", "Plot generated"),
        ("plot-voi-curves", "curves", "Plot generated"),
        ("plot-dominance", "dominance", "Plot generated"),
    ],
)
def test_cli_plot_commands_e2e(
    command_name: str,
    setup: str,
    expected_output: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-test the plotting commands through the Typer CLI."""
    output_file = tmp_path / f"{setup}.png"

    if setup == "ceac":
        surface_file = tmp_path / "surface.json"
        _write_json(
            surface_file,
            {
                "strategy_names": ["Strategy A", "Strategy B"],
                "wtp_thresholds": [0, 50000],
                "net_benefit": [
                    [[10, 15], [12, 14]],
                    [[11, 16], [13, 15]],
                ],
            },
        )
        _patch_plot_renderer(monkeypatch, "render_ceac")
        args = [
            "--format",
            "json",
            command_name,
            str(surface_file),
            "--output",
            str(output_file),
        ]
    elif setup == "ceaf":
        surface_file = tmp_path / "surface.json"
        _write_json(
            surface_file,
            {
                "strategy_names": ["Strategy A", "Strategy B"],
                "wtp_thresholds": [0, 50000],
                "net_benefit": [
                    [[10, 15], [12, 14]],
                    [[11, 16], [13, 15]],
                ],
            },
        )
        _patch_plot_renderer(monkeypatch, "render_ceaf")
        monkeypatch.setattr(cli, "calculate_ceaf_result", lambda *args, **kwargs: 1.0)
        args = [
            "--format",
            "json",
            command_name,
            str(surface_file),
            "--output",
            str(output_file),
        ]
    elif setup == "curves":
        series_file = tmp_path / "curves.json"
        _write_json(
            series_file,
            {"wtp_thresholds": [0, 1, 2], "evpi_values": [1.0, 1.5, 2.0]},
        )
        _patch_plot_renderer(monkeypatch, "render_evpi_vs_wtp")
        args = [
            "--format",
            "json",
            command_name,
            str(series_file),
            "--output",
            str(output_file),
        ]
    elif setup == "dominance":
        input_file = tmp_path / "dominance.csv"
        input_file.write_text(
            "strategy,cost,effect\nA,100,1.0\nB,150,1.2\nC,130,1.1\n",
            encoding="utf-8",
        )
        _patch_plot_renderer(monkeypatch, "render_dominance")
        monkeypatch.setattr(
            cli, "calculate_dominance_result", lambda *args, **kwargs: {}
        )
        args = [
            "--format",
            "json",
            command_name,
            str(input_file),
            "--output",
            str(output_file),
        ]
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unhandled setup: {setup}")

    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == command_name
    assert payload["saved"] is True
    if command_name in {
        "calculate-evpi",
        "calculate-evppi",
        "calculate-evsi",
        "calculate-enbs",
    }:
        assert payload["reporting"]["reporting_standard"] == "CHEERS-VOI"
    assert expected_output == "Plot generated"
    assert output_file.exists()


def test_cli_generate_config_e2e(tmp_path: Path) -> None:
    """Smoke-test config generation."""
    output_file = tmp_path / "evsi_config.json"
    result = runner.invoke(
        cli.app,
        ["generate-config", "evsi", "--output", str(output_file)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "calculate-evsi"
    assert payload["method"] == "two_loop"
    assert output_file.exists()


def test_cli_e2e_error_paths(tmp_path: Path) -> None:
    """Smoke-test representative CLI error handling."""
    missing_file = tmp_path / "missing.csv"
    missing_result = runner.invoke(cli.app, ["calculate-evpi", str(missing_file)])
    assert missing_result.exit_code != 0
    assert "Error" in missing_result.stderr

    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{", encoding="utf-8")
    invalid_result = runner.invoke(cli.app, ["calculate-nma-voi", str(invalid_config)])
    assert invalid_result.exit_code != 0
    assert "Error" in invalid_result.stderr
