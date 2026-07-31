"""User-surface contracts for experimental deterministic sensitivity analysis."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from voiage.cli import app
from voiage.methods.deterministic_sensitivity import (
    deterministic_sensitivity_from_specification,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/deterministic-sensitivity-analysis/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _input() -> dict[str, Any]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def test_dsa_cli_returns_exact_versioned_result_and_writes_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dsa-result.json"
    response = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-deterministic-sensitivity",
            str(INPUT),
            "--output",
            str(output),
        ],
    )

    assert response.exit_code == 0, response.stdout
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert json.loads(response.stdout) == expected
    assert json.loads(output.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload["tie_tolerance"].update({"unknown": True}),
        lambda payload: payload["baseline"][0].pop("unit"),
        lambda payload: payload["model_evaluation_records"][0]["alternative_outputs"][
            0
        ].update({"unit": "wrong-unit"}),
    ],
)
def test_dsa_installed_runtime_validation_is_fail_closed(mutate: Any) -> None:
    payload = deepcopy(_input())
    mutate(payload)

    with pytest.raises(ValueError):
        deterministic_sensitivity_from_specification(payload)


def test_dsa_cli_rejects_invalid_json_object(tmp_path: Path) -> None:
    request = tmp_path / "invalid.json"
    request.write_text('{"schema_version": "wrong"}', encoding="utf-8")

    response = CliRunner().invoke(
        app, ["calculate-deterministic-sensitivity", str(request)]
    )

    assert response.exit_code == 1
    assert "Error:" in response.stderr


def test_dsa_tornado_plot_uses_grid_extrema_units_and_rank_order() -> None:
    from voiage.plot import plot_deterministic_sensitivity_tornado

    result = deterministic_sensitivity_from_specification(_input())
    ax = plot_deterministic_sensitivity_tornado(result)

    assert len(ax.patches) == len(result.parameter_summaries)
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        summary.parameter_name for summary in result.parameter_summaries
    ]
    assert ax.get_xlabel() == "Optimal metric (net-benefit-point)"
    assert any(line.get_label() == "Baseline optimum" for line in ax.lines)
    assert all(patch.get_hatch() for patch in ax.patches)


def test_dsa_tornado_plot_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voiage.plot import deterministic_sensitivity as plotting

    monkeypatch.setattr(plotting, "MATPLOTLIB_AVAILABLE", False)
    result = deterministic_sensitivity_from_specification(_input())

    with pytest.raises(Exception, match="Matplotlib is required"):
        plotting.plot_deterministic_sensitivity_tornado(result)
