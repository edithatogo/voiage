"""Focused CLI coverage for calibration and observational commands."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from voiage import cli

runner = CliRunner()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_parameter_csv(path: Path) -> None:
    path.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")


def _json_payload(stdout: str) -> dict[str, object]:
    return json.loads(stdout)


def _assert_missing_file(result: object) -> None:
    assert result.exit_code == 2


def test_calibration_cli_happy_path_and_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise calibration CLI JSON payload, fallback model, and file output."""
    parameter_file = tmp_path / "parameters.csv"
    study_file = tmp_path / "calibration_study.json"
    process_file = tmp_path / "calibration_process.json"
    output_file = tmp_path / "calibration.txt"
    _write_parameter_csv(parameter_file)
    _write_json(study_file, {"study_type": "lab", "sample_size": 10})
    _write_json(process_file, {"update_rule": "bayesian"})

    seen: dict[str, object] = {}

    def fake_read_parameter_set_csv(path: str, skip_header: bool = False) -> object:
        assert path == str(parameter_file)
        assert skip_header is True
        return SimpleNamespace(tag="psa")

    def fake_voi_calibration(**kwargs: object) -> float:
        seen.update(kwargs)
        assert kwargs["cal_study_modeler"] is None
        assert kwargs["psa_prior"].tag == "psa"
        assert kwargs["calibration_study_design"] == {
            "study_type": "lab",
            "sample_size": 10,
        }
        assert kwargs["calibration_process_spec"] == {"update_rule": "bayesian"}
        assert kwargs["population"] == 1000.0
        assert kwargs["discount_rate"] == 0.03
        assert kwargs["time_horizon"] == 5.0
        assert kwargs["n_outer_loops"] == 7
        return 3.25

    monkeypatch.setattr(cli, "read_parameter_set_csv", fake_read_parameter_set_csv)
    monkeypatch.setattr(cli, "voi_calibration", fake_voi_calibration)

    json_result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "calculate-calibration",
            str(parameter_file),
            str(study_file),
            str(process_file),
            "--population",
            "1000",
            "--discount-rate",
            "0.03",
            "--time-horizon",
            "5",
            "--n-outer-loops",
            "7",
        ],
    )

    assert json_result.exit_code == 0
    payload = _json_payload(json_result.stdout)
    assert payload["command"] == "calculate-calibration"
    assert payload["metric"] == "Calibration VOI"
    assert payload["value"] == pytest.approx(3.25)
    assert payload["reporting"]["reporting_standard"] == "CHEERS-VOI"
    assert payload["reporting"]["analysis_type"] == "calculate-calibration"
    assert payload["reporting"]["method_family"] == "calibration"
    assert payload["reporting"]["estimator"] == "sophisticated"
    assert payload["reporting"]["reproducibility"]["n_outer_loops"] == 7
    assert seen["cal_study_modeler"] is None

    file_result = runner.invoke(
        cli.app,
        [
            "calculate-calibration",
            str(parameter_file),
            str(study_file),
            str(process_file),
            "--output",
            str(output_file),
            "--population",
            "1000",
            "--discount-rate",
            "0.03",
            "--time-horizon",
            "5",
            "--n-outer-loops",
            "7",
        ],
    )

    assert file_result.exit_code == 0
    assert "Calibration VOI: 3.250000" in file_result.stdout
    assert f"Result saved to {output_file}" in file_result.stdout
    assert output_file.read_text(encoding="utf-8").strip() == (
        "Calibration VOI: 3.250000"
    )


def test_calibration_cli_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise calibration CLI missing-file and non-object JSON branches."""
    parameter_file = tmp_path / "parameters.csv"
    study_file = tmp_path / "calibration_study.json"
    process_file = tmp_path / "calibration_process.json"
    _write_parameter_csv(parameter_file)
    _write_json(study_file, [])
    _write_json(process_file, {"update_rule": "bayesian"})

    monkeypatch.setattr(
        cli,
        "read_parameter_set_csv",
        lambda path, skip_header=False: SimpleNamespace(tag="psa"),
    )

    invalid_result = runner.invoke(
        cli.app,
        [
            "calculate-calibration",
            str(parameter_file),
            str(study_file),
            str(process_file),
        ],
    )
    assert invalid_result.exit_code == 1
    assert "Calibration study design file must contain a JSON object" in (
        invalid_result.stderr
    )

    missing_result = runner.invoke(
        cli.app,
        [
            "calculate-calibration",
            str(tmp_path / "missing.csv"),
            str(study_file),
            str(process_file),
        ],
    )
    _assert_missing_file(missing_result)


def test_observational_cli_happy_path_and_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise observational CLI JSON payload, fallback model, and file output."""
    parameter_file = tmp_path / "parameters.csv"
    study_file = tmp_path / "observational_study.json"
    bias_file = tmp_path / "bias_models.json"
    output_file = tmp_path / "observational.txt"
    _write_parameter_csv(parameter_file)
    _write_json(study_file, {"study_type": "cohort", "sample_size": 25})
    _write_json(bias_file, {"bias_type": "measurement"})

    seen: dict[str, object] = {}

    def fake_read_parameter_set_csv(path: str, skip_header: bool = False) -> object:
        assert path == str(parameter_file)
        assert skip_header is True
        return SimpleNamespace(tag="psa")

    def fake_voi_observational(**kwargs: object) -> float:
        seen.update(kwargs)
        assert kwargs["obs_study_modeler"] is None
        assert kwargs["psa_prior"].tag == "psa"
        assert kwargs["observational_study_design"] == {
            "study_type": "cohort",
            "sample_size": 25,
        }
        assert kwargs["bias_models"] == {"bias_type": "measurement"}
        assert kwargs["population"] == 1500.0
        assert kwargs["discount_rate"] == 0.02
        assert kwargs["time_horizon"] == 4.0
        assert kwargs["n_outer_loops"] == 6
        return 4.5

    monkeypatch.setattr(cli, "read_parameter_set_csv", fake_read_parameter_set_csv)
    monkeypatch.setattr(cli, "voi_observational", fake_voi_observational)

    json_result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "calculate-observational",
            str(parameter_file),
            str(study_file),
            str(bias_file),
            "--population",
            "1500",
            "--discount-rate",
            "0.02",
            "--time-horizon",
            "4",
            "--n-outer-loops",
            "6",
        ],
    )

    assert json_result.exit_code == 0
    payload = _json_payload(json_result.stdout)
    assert payload["command"] == "calculate-observational"
    assert payload["metric"] == "Observational VOI"
    assert payload["value"] == pytest.approx(4.5)
    assert payload["reporting"]["reporting_standard"] == "CHEERS-VOI"
    assert payload["reporting"]["analysis_type"] == "calculate-observational"
    assert payload["reporting"]["method_family"] == "observational"
    assert payload["reporting"]["estimator"] == "basic"
    assert payload["reporting"]["reproducibility"]["n_outer_loops"] == 6
    assert seen["obs_study_modeler"] is None

    file_result = runner.invoke(
        cli.app,
        [
            "calculate-observational",
            str(parameter_file),
            str(study_file),
            str(bias_file),
            "--output",
            str(output_file),
            "--population",
            "1500",
            "--discount-rate",
            "0.02",
            "--time-horizon",
            "4",
            "--n-outer-loops",
            "6",
        ],
    )

    assert file_result.exit_code == 0
    assert "Observational VOI: 4.500000" in file_result.stdout
    assert f"Result saved to {output_file}" in file_result.stdout
    assert output_file.read_text(encoding="utf-8").strip() == (
        "Observational VOI: 4.500000"
    )


def test_observational_cli_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise observational CLI missing-file and non-object JSON branches."""
    parameter_file = tmp_path / "parameters.csv"
    study_file = tmp_path / "observational_study.json"
    bias_file = tmp_path / "bias_models.json"
    _write_parameter_csv(parameter_file)
    _write_json(study_file, {})
    _write_json(bias_file, [])

    monkeypatch.setattr(
        cli,
        "read_parameter_set_csv",
        lambda path, skip_header=False: SimpleNamespace(tag="psa"),
    )

    invalid_result = runner.invoke(
        cli.app,
        [
            "calculate-observational",
            str(parameter_file),
            str(study_file),
            str(bias_file),
        ],
    )
    assert invalid_result.exit_code == 1
    assert "Bias models file must contain a JSON object" in invalid_result.stderr

    missing_result = runner.invoke(
        cli.app,
        [
            "calculate-observational",
            str(parameter_file),
            str(tmp_path / "missing_observational.json"),
            str(bias_file),
        ],
    )
    _assert_missing_file(missing_result)
