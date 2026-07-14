"""Branch-focused CLI tests for plotting commands."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from voiage import cli

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _write_perspective_surface(path: Path) -> None:
    _write_json(
        path,
        {
            "net_benefit": [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ],
            "strategy_names": ["A", "B"],
            "perspective_names": ["payer", "societal"],
            "perspective_weights": {"payer": 0.4, "societal": 0.6},
            "reference_perspective": "payer",
        },
    )


def _write_surface(path: Path) -> None:
    _write_json(
        path,
        {
            "strategy_names": ["A", "B"],
            "wtp_thresholds": [0.0, 50000.0],
            "net_benefit": [
                [[10.0, 15.0], [12.0, 14.0]],
                [[11.0, 16.0], [13.0, 15.0]],
            ],
        },
    )


def _write_evpi_series(path: Path) -> None:
    _write_json(path, {"wtp_thresholds": [0.0, 1.0], "evpi_values": [1.0, 2.0]})


def _write_evsi_series(path: Path) -> None:
    _write_json(
        path,
        {
            "sample_sizes": [10.0, 20.0],
            "evsi_values": [0.5, 0.75],
            "enbs_values": [1.0, 1.5],
            "research_costs": [0.1, 0.2],
        },
    )


def _write_dominance_csv(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "cost", "effect"])
        writer.writerow(["A", 100, 1.0])
        writer.writerow(["B", 150, 1.2])


def _patch_renderer(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(cli, name, lambda *args, **kwargs: _DummyAxes())


@pytest.mark.parametrize(
    (
        "command_name",
        "setup",
        "renderer_name",
        "input_filename",
        "writer",
        "extra_patches",
    ),
    [
        (
            "plot-perspective-regret",
            "perspective",
            "render_perspective_regret",
            "perspective.json",
            _write_perspective_surface,
            (),
        ),
        (
            "plot-ceac",
            "surface",
            "render_ceac",
            "surface.json",
            _write_surface,
            (),
        ),
        (
            "plot-ceaf",
            "surface",
            "render_ceaf",
            "surface.json",
            _write_surface,
            (("calculate_ceaf_result", lambda *args, **kwargs: object()),),
        ),
        (
            "plot-voi-curves",
            "evpi",
            "render_evpi_vs_wtp",
            "series.json",
            _write_evpi_series,
            (),
        ),
        (
            "plot-voi-curves",
            "evsi",
            "render_evsi_vs_sample_size",
            "series.json",
            _write_evsi_series,
            (),
        ),
        (
            "plot-dominance",
            "dominance",
            "render_dominance",
            "dominance.csv",
            _write_dominance_csv,
            (("calculate_dominance_result", lambda *args, **kwargs: object()),),
        ),
    ],
)
def test_plot_commands_json_output_branch(
    command_name: str,
    setup: str,
    renderer_name: str,
    input_filename: str,
    writer: Callable[[Path], None],
    extra_patches: tuple[tuple[str, object], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plot commands should emit JSON output and save figures when requested."""
    input_file = tmp_path / input_filename
    writer(input_file)
    output_file = tmp_path / f"{setup}.png"
    _patch_renderer(monkeypatch, renderer_name)
    for attr_name, value in extra_patches:
        monkeypatch.setattr(cli, attr_name, value)

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            command_name,
            str(input_file),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == command_name
    assert payload["saved"] is True
    assert output_file.read_text(encoding="utf-8") == "figure"


@pytest.mark.parametrize(
    (
        "command_name",
        "setup",
        "renderer_name",
        "input_filename",
        "writer",
        "extra_patches",
    ),
    [
        (
            "plot-perspective-regret",
            "perspective",
            "render_perspective_regret",
            "perspective.json",
            _write_perspective_surface,
            (),
        ),
        (
            "plot-ceac",
            "surface",
            "render_ceac",
            "surface.json",
            _write_surface,
            (),
        ),
        (
            "plot-ceaf",
            "surface",
            "render_ceaf",
            "surface.json",
            _write_surface,
            (("calculate_ceaf_result", lambda *args, **kwargs: object()),),
        ),
        (
            "plot-voi-curves",
            "evpi",
            "render_evpi_vs_wtp",
            "series.json",
            _write_evpi_series,
            (),
        ),
        (
            "plot-voi-curves",
            "evsi",
            "render_evsi_vs_sample_size",
            "series.json",
            _write_evsi_series,
            (),
        ),
        (
            "plot-dominance",
            "dominance",
            "render_dominance",
            "dominance.csv",
            _write_dominance_csv,
            (("calculate_dominance_result", lambda *args, **kwargs: object()),),
        ),
    ],
)
def test_plot_commands_quiet_output_branch(
    command_name: str,
    setup: str,
    renderer_name: str,
    input_filename: str,
    writer: Callable[[Path], None],
    extra_patches: tuple[tuple[str, object], ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plot commands should still save figures under quiet mode."""
    input_file = tmp_path / input_filename
    writer(input_file)
    output_file = tmp_path / f"{setup}.png"
    _patch_renderer(monkeypatch, renderer_name)
    for attr_name, value in extra_patches:
        monkeypatch.setattr(cli, attr_name, value)

    result = runner.invoke(
        cli.app,
        [
            "--quiet",
            command_name,
            str(input_file),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert "Plot generated" in result.stdout
    assert "Plot saved to" not in result.stdout
    assert output_file.read_text(encoding="utf-8") == "figure"


@pytest.mark.parametrize(
    ("command_name", "setup", "patch_name", "error", "message"),
    [
        (
            "plot-perspective-regret",
            "perspective",
            "_read_perspective_surface",
            FileNotFoundError("missing perspective surface"),
            "Error: File not found - missing perspective surface",
        ),
        (
            "plot-ceac",
            "surface",
            "_read_plot_surface",
            TypeError("bad surface"),
            "Error: bad surface",
        ),
        (
            "plot-ceaf",
            "surface",
            "calculate_ceaf_result",
            ValueError("bad ceaf"),
            "Error: bad ceaf",
        ),
        (
            "plot-voi-curves",
            "evsi",
            "_read_plot_series",
            ValueError("bad series"),
            "Error: bad series",
        ),
        (
            "plot-dominance",
            "dominance",
            "_read_cost_effect_csv",
            RuntimeError("boom"),
            "An error occurred: boom",
        ),
    ],
)
def test_plot_commands_error_handling(
    command_name: str,
    setup: str,
    patch_name: str,
    error: Exception,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plot commands should surface the expected error handler branches."""
    input_file = tmp_path / f"{setup}.json"
    if command_name == "plot-dominance":
        input_file = tmp_path / "dominance.csv"
        _write_dominance_csv(input_file)
    elif command_name == "plot-voi-curves" and setup == "evsi":
        _write_evsi_series(input_file)
    elif command_name == "plot-perspective-regret":
        _write_perspective_surface(input_file)
    else:
        _write_surface(input_file)

    def _raise(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(cli, patch_name, _raise)

    result = runner.invoke(
        cli.app,
        [command_name, str(input_file), "--output", str(tmp_path / "out.png")],
    )

    assert result.exit_code == 1
    assert message in result.stderr
