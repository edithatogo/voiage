"""User-facing CLI, reporting, and plotting checks for experimental COSS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from typer.testing import CliRunner

from voiage import cli
from voiage.contracts.study_design import CossPlotDataV1, CossResultV1
from voiage.plot.voi_curves import plot_coss

runner = CliRunner()
EXAMPLE = (
    Path(__file__).parents[1] / "examples" / "cli_samples" / "coss_study_design.json"
)


def _calculate_example() -> dict[str, object]:
    response = runner.invoke(
        cli.app,
        ["--format", "json", "calculate-coss", str(EXAMPLE)],
    )
    assert response.exit_code == 0, response.output
    return json.loads(response.stdout)


def test_calculate_coss_cli_emits_contract_reporting_and_provenance() -> None:
    payload = _calculate_example()

    assert payload["method_maturity"] == "experimental"
    result = payload["result"]
    assert result["schema_version"] == "1.0.0"
    assert result["optimal_design_id"] == "n-200"
    assert result["maximum_enbs"] == 60000.0
    reporting = payload["reporting"]
    assert reporting["method_maturity"] == "experimental"
    assert reporting["decision_problem_id"] == "example-treatment-choice"
    assert reporting["provenance"]["runtime"] == "rust"
    assert reporting["reproducibility"]["value_unit"] == "AUD-2026"


def test_calculate_coss_output_can_be_plotted_without_reshaping(tmp_path: Path) -> None:
    result_path = tmp_path / "coss-result.json"
    image_path = tmp_path / "coss.png"
    result_path.write_text(json.dumps(_calculate_example()), encoding="utf-8")

    response = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "plot-coss",
            str(result_path),
            "--output",
            str(image_path),
        ],
    )

    assert response.exit_code == 0, response.output
    assert json.loads(response.stdout)["saved"] is True
    assert image_path.stat().st_size > 0


def test_plot_coss_accepts_plot_contract_and_uses_redundant_encodings() -> None:
    data = CossPlotDataV1(
        design_ids=("d-1", "d-2", "d-3"),
        sample_sizes=(100, 200, 300),
        evsi=(10.0, 20.0, 24.0),
        research_cost=(8.0, 12.0, 20.0),
        enbs=(2.0, 8.0, 4.0),
        feasible=(True, True, False),
        enbs_lower=(1.0, 6.0, None),
        enbs_upper=(3.0, 10.0, None),
        optimal_design_id="d-2",
        tied_optimal_design_ids=("d-2",),
        boundary_state="upper",
    )

    figure, existing_axis = plt.subplots()
    axis = plot_coss(data, ax=existing_axis)

    assert axis is existing_axis
    labels = axis.get_legend_handles_labels()[1]
    assert "Feasible design" in labels
    assert "Infeasible design" in labels
    assert "Selected optimum (d-2)" in labels
    assert "ENBS uncertainty interval" in labels
    plt.close(figure)


def test_plot_coss_exposes_complete_ties_and_unavailable_selection_uncertainty() -> None:
    payload = _calculate_example()["result"]
    assert isinstance(payload, dict)
    result = CossResultV1.model_validate_json(json.dumps(payload))
    tied_data = result.plot_data.model_copy(
        update={"tied_optimal_design_ids": ("n-100", "n-200")}
    )

    figure, axis = plt.subplots()
    try:
        _ = plot_coss(tied_data, ax=axis)
        labels = axis.get_legend_handles_labels()[1]
        assert "Tied optima (n-100, n-200)" in labels
    finally:
        plt.close(figure)

    figure, axis = plt.subplots()
    try:
        _ = plot_coss(result, ax=axis)
        labels = axis.get_legend_handles_labels()[1]
        assert "Selection uncertainty unavailable" in labels
    finally:
        plt.close(figure)
