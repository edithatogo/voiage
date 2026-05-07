"""Focused CLI tests for sequential and NMA VOI branches."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voiage import cli

runner = CliRunner()


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_calculate_sequential_voi_rejects_missing_time_steps(
    tmp_path: Path,
) -> None:
    """Sequential CLI should reject a dynamic spec without time_steps."""
    parameter_file = tmp_path / "parameters.csv"
    _write_csv(parameter_file, [["param1"], [0.1], [0.2]])

    dynamic_spec_file = tmp_path / "dynamic_spec.json"
    _write_json(dynamic_spec_file, {"discount_rate": 0.03})

    result = runner.invoke(
        cli.app,
        [
            "calculate-sequential-voi",
            str(parameter_file),
            str(dynamic_spec_file),
        ],
    )

    assert result.exit_code == 1
    assert "time_steps" in result.stderr


def test_calculate_sequential_voi_rejects_non_numeric_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequential CLI should surface a non-numeric result from the estimator."""
    parameter_file = tmp_path / "parameters.csv"
    _write_csv(parameter_file, [["param1"], [0.1], [0.2]])

    dynamic_spec_file = tmp_path / "dynamic_spec.json"
    _write_json(dynamic_spec_file, {"time_steps": [0, 1, 2]})

    monkeypatch.setattr(cli, "sequential_voi", lambda **kwargs: object())

    result = runner.invoke(
        cli.app,
        [
            "calculate-sequential-voi",
            str(parameter_file),
            str(dynamic_spec_file),
        ],
    )

    assert result.exit_code == 1
    assert "Sequential VOI did not return a numeric result" in result.stderr


def test_calculate_sequential_voi_writes_output_file_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequential CLI should write the result file and emit the save message."""
    parameter_file = tmp_path / "parameters.csv"
    _write_csv(parameter_file, [["param1"], [0.1], [0.2]])

    dynamic_spec_file = tmp_path / "dynamic_spec.json"
    _write_json(dynamic_spec_file, {"time_steps": [0, 1, 2]})

    monkeypatch.setattr(cli, "sequential_voi", lambda **kwargs: 2.75)

    output_file = tmp_path / "sequential.txt"
    result = runner.invoke(
        cli.app,
        [
            "calculate-sequential-voi",
            str(parameter_file),
            str(dynamic_spec_file),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "Sequential VOI: 2.750000" in result.stdout
    assert f"Result saved to {output_file}" in result.stdout
    assert output_file.read_text(encoding="utf-8").strip() == "Sequential VOI: 2.750000"


def test_calculate_nma_voi_evpi_branch_writes_output_file_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NMA CLI should take the EVPI branch when no parameters are provided."""
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

    seen: dict[str, object] = {}

    def fake_evpi(nma_data: object, **kwargs: object) -> float:
        seen["nma_data"] = nma_data
        seen["kwargs"] = kwargs
        return 6.5

    monkeypatch.setattr(cli, "calculate_nma_evpi", fake_evpi)

    output_file = tmp_path / "nma_evpi.txt"
    result = runner.invoke(
        cli.app,
        [
            "calculate-nma-voi",
            str(config_file),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "NMA-EVPI: 6.500000" in result.stdout
    assert f"Result saved to {output_file}" in result.stdout
    assert output_file.read_text(encoding="utf-8").strip() == "NMA-EVPI: 6.500000"
    assert seen["kwargs"]["willingness_to_pay"] is None


def test_calculate_nma_voi_evppi_branch_uses_parameter_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NMA CLI should take the EVPPI branch and build parameter samples."""
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

    monkeypatch.setattr(
        cli.np.random,
        "rand",
        lambda n_samples: cli.np.full(n_samples, 0.25, dtype=float),
    )

    seen: dict[str, object] = {}

    def fake_evppi(nma_data: object, **kwargs: object) -> float:
        seen["nma_data"] = nma_data
        seen["kwargs"] = kwargs
        return 7.5

    monkeypatch.setattr(cli, "calculate_nma_evppi", fake_evppi)

    output_file = tmp_path / "nma_evppi.txt"
    result = runner.invoke(
        cli.app,
        [
            "calculate-nma-voi",
            str(config_file),
            "--parameters-of-interest",
            "effect_A,effect_B",
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "NMA-EVPPI: 7.500000" in result.stdout
    assert f"Result saved to {output_file}" in result.stdout
    assert output_file.read_text(encoding="utf-8").strip() == "NMA-EVPPI: 7.500000"
    assert seen["kwargs"]["parameters_of_interest"] == ["effect_A", "effect_B"]
    assert set(seen["kwargs"]["parameter_samples"]) == {"effect_A", "effect_B"}
