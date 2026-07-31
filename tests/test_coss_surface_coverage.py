"""Focused behavioural coverage for COSS runtime, CLI, and plotting surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from typer.testing import CliRunner

from voiage import _runtime, cli
from voiage.contracts.study_design import CossPlotDataV1
from voiage.exceptions import InputError, PlottingError
from voiage.plot import voi_curves

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cli_samples" / "coss_study_design.json"
runner = CliRunner()


def _example_payload() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _plot_data(*, feasible: tuple[bool, ...]) -> CossPlotDataV1:
    return CossPlotDataV1(
        design_ids=("n-100", "n-200"),
        sample_sizes=(100, 200),
        evsi=(110.0, 160.0),
        research_cost=(90.0, 130.0),
        enbs=(20.0, 30.0),
        feasible=feasible,
        enbs_lower=(None, None),
        enbs_upper=(None, None),
        boundary_state="none",
    )


def test_compute_coss_translates_native_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeInputError(Exception):
        category = "input"
        diagnostic_code = "invalid_coss"

    class FailingNative:
        def compute_coss(self, *args: object) -> object:
            raise NativeInputError("invalid COSS input")

    native = FailingNative()

    def load_failing_native() -> FailingNative:
        return native

    monkeypatch.setattr(_runtime, "_native", load_failing_native)

    with pytest.raises(InputError, match="invalid COSS input") as raised:
        _runtime.compute_coss(
            sample_sizes=[100],
            evsi_values=[10.0],
            research_costs=[5.0],
            feasible=[True],
            tie_policy="smallest_sample_size",
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
        )

    assert raised.value.diagnostic_code == "invalid_coss"


def test_coss_readers_reject_non_object_and_empty_designs(tmp_path: Path) -> None:
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="COSS specification"):
        _ = cli._read_coss_specification(non_object)
    with pytest.raises(TypeError, match="COSS result"):
        _ = cli._read_coss_result(non_object)

    empty_designs = _example_payload()
    empty_designs["designs"] = []
    empty_path = tmp_path / "empty-designs.json"
    empty_path.write_text(json.dumps(empty_designs), encoding="utf-8")
    with pytest.raises(TypeError, match="non-empty list"):
        _ = cli._read_coss_specification(empty_path)


def test_calculate_coss_cli_writes_text_result_and_status(tmp_path: Path) -> None:
    output = tmp_path / "coss.txt"
    response = runner.invoke(
        cli.app,
        ["calculate-coss", str(EXAMPLE), "--output", str(output)],
    )

    assert response.exit_code == 0, response.output
    assert "COSS optimum: n-200" in response.output
    assert f"Result saved to {output}" in response.output
    assert output.read_text(encoding="utf-8").startswith("COSS optimum: n-200")


def test_calculate_coss_cli_reports_no_feasible_design(tmp_path: Path) -> None:
    payload = _example_payload()
    for design in payload["designs"]:
        design["feasible"] = False
    specification = tmp_path / "no-feasible.json"
    specification.write_text(json.dumps(payload), encoding="utf-8")

    response = runner.invoke(cli.app, ["calculate-coss", str(specification)])

    assert response.exit_code == 0, response.output
    assert response.output.strip() == "COSS optimum: unavailable (no feasible design)"


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (FileNotFoundError("missing COSS specification"), "File not found"),
        (RuntimeError("unexpected COSS failure"), "An error occurred"),
    ],
)
def test_calculate_coss_cli_reports_reader_and_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected: str,
) -> None:
    if isinstance(failure, FileNotFoundError):
        monkeypatch.setattr(
            cli,
            "_read_coss_specification",
            lambda _path: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(
            cli,
            "calculate_coss_result",
            lambda **_kwargs: (_ for _ in ()).throw(failure),
        )

    response = runner.invoke(cli.app, ["calculate-coss", str(EXAMPLE)])

    assert response.exit_code == 1
    assert expected in response.output


def test_calculate_coss_cli_reports_invalid_json_shape(tmp_path: Path) -> None:
    specification = tmp_path / "invalid-shape.json"
    specification.write_text("[]", encoding="utf-8")

    response = runner.invoke(cli.app, ["calculate-coss", str(specification)])

    assert response.exit_code == 1
    assert "COSS specification must contain a JSON object" in response.output


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (FileNotFoundError("missing COSS result"), "File not found"),
        (RuntimeError("unexpected renderer failure"), "An error occurred"),
    ],
)
def test_plot_coss_cli_reports_reader_and_renderer_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected: str,
) -> None:
    if isinstance(failure, FileNotFoundError):
        monkeypatch.setattr(
            cli,
            "_read_coss_result",
            lambda _path: (_ for _ in ()).throw(failure),
        )
    else:
        monkeypatch.setattr(cli, "_read_coss_result", lambda _path: object())
        monkeypatch.setattr(
            cli,
            "render_coss",
            lambda _result: (_ for _ in ()).throw(failure),
        )

    response = runner.invoke(cli.app, ["plot-coss", str(EXAMPLE)])

    assert response.exit_code == 1
    assert expected in response.output


def test_plot_coss_cli_reports_invalid_result_json(tmp_path: Path) -> None:
    result = tmp_path / "invalid-result.json"
    result.write_text("[]", encoding="utf-8")

    response = runner.invoke(cli.app, ["plot-coss", str(result)])

    assert response.exit_code == 1
    assert "COSS result must contain a JSON object" in response.output


def test_plot_coss_cli_reports_generated_plot_without_saving(tmp_path: Path) -> None:
    calculation = runner.invoke(
        cli.app,
        ["--format", "json", "calculate-coss", str(EXAMPLE)],
    )
    assert calculation.exit_code == 0, calculation.output
    result = tmp_path / "result.json"
    result.write_text(calculation.output, encoding="utf-8")

    response = runner.invoke(cli.app, ["plot-coss", str(result)])

    assert response.exit_code == 0, response.output
    assert response.output.strip() == "Plot generated"


def test_plot_coss_reports_dependency_and_input_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voi_curves, "MATPLOTLIB_AVAILABLE", False)
    with pytest.raises(PlottingError, match="Matplotlib is required"):
        _ = voi_curves.plot_coss(_plot_data(feasible=(True, True)))

    monkeypatch.setattr(voi_curves, "MATPLOTLIB_AVAILABLE", True)
    with pytest.raises(InputError, match="CossResultV1 or CossPlotDataV1"):
        _ = voi_curves.plot_coss(object())


@pytest.mark.parametrize("feasible", [(True, True), (False, False)])
def test_plot_coss_handles_absent_optional_encodings(
    feasible: tuple[bool, ...],
) -> None:
    figure, axis = plt.subplots()
    try:
        result = voi_curves.plot_coss(_plot_data(feasible=feasible), ax=axis)
        assert result is axis
        labels = axis.get_legend_handles_labels()[1]
        assert "ENBS uncertainty interval" not in labels
        assert not any("Selected optimum" in label for label in labels)
    finally:
        plt.close(figure)
