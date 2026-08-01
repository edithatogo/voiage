"""CLI, plotting and public-discovery assurance for issue #560."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.methods.mcda_information import (
    McdaInformationResult,
    mcda_information_value,
)
from voiage.plot import plot_mcda_information_value, plot_mcda_rank_acceptability

matplotlib.use("Agg")

FIXTURE = "specs/frontier/mcda-information/v1/fixtures/normative/input.json"


def _result() -> McdaInformationResult:
    payload = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    return mcda_information_value(payload)


def test_cli_returns_versioned_json_and_writes_identical_output(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-mcda-information",
            FIXTURE,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["analysis_type"] == "mcda_perfect_information_result"
    assert payload["method_maturity"] == "experimental"
    assert payload["decomposition"]["joint_gross_voi"] == pytest.approx(0.028)
    assert payload["decomposition"]["interaction"] == pytest.approx(0.028)
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_cli_text_output_names_every_information_component(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    result = CliRunner().invoke(
        app,
        ["calculate-mcda-information", FIXTURE, "--output", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    for phrase in (
        "joint 0.028000",
        "criterion 0.000000",
        "preference 0.000000",
        "interaction 0.028000",
    ):
        assert phrase in result.stdout
    assert "normalized decision value per eligible person" in result.stdout
    assert output.read_text(encoding="utf-8").startswith("MCDA information value:")


def test_cli_rejects_malformed_non_object_and_semantically_invalid_requests(
    tmp_path: Path,
) -> None:
    valid = Path(FIXTURE).read_text(encoding="utf-8")
    for index, content in enumerate(
        [
            "{",
            "[]",
            "{}",
            valid.replace('"probability": 0.35', '"probability": -0.35', 1),
        ]
    ):
        request = tmp_path / f"bad-{index}.json"
        request.write_text(content, encoding="utf-8")
        result = CliRunner().invoke(app, ["calculate-mcda-information", str(request)])
        assert result.exit_code == 1
        assert "Error:" in result.output


def test_experimental_public_exports_preserve_identity() -> None:
    assert voiage.McdaInformationResult is McdaInformationResult
    assert voiage.mcda_information_value is mcda_information_value


def test_information_plot_has_labels_hatches_and_numeric_annotations() -> None:
    ax = plot_mcda_information_value(_result())
    assert ax.get_xlabel() == "Resolved uncertainty"
    assert "Gross information value" in ax.get_ylabel()
    assert [patch.get_hatch() for patch in ax.patches]
    assert len(ax.texts) == 3
    plt.close(ax.figure)


def test_rank_plot_has_explicit_markers_labels_and_probability_axis() -> None:
    ax = plot_mcda_rank_acceptability(_result())
    assert ax.get_xlabel() == "Rank"
    assert ax.get_ylabel() == "Probability (fractional ties)"
    assert all(line.get_marker() for line in ax.lines)
    assert ax.get_legend() is not None
    plt.close(ax.figure)
