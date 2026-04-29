"""Tests for Value of Perspective analysis."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from typer.testing import CliRunner

from voiage import cli
from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.perspective import (
    Perspective,
    PerspectiveSet,
    ValueOfPerspectiveResult,
    perspective_optimal_strategies,
    value_of_perspective,
)
from voiage.plot.perspective import plot_perspective_regret
from voiage.schema import ValueArray

runner = CliRunner()


class _DummyFigure:
    def savefig(self, output_file: Path, bbox_inches: str = "tight") -> None:
        _ = bbox_inches
        Path(output_file).write_text("figure", encoding="utf-8")


class _DummyAxes:
    def __init__(self) -> None:
        self.figure = _DummyFigure()


def _write_perspective_surface(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "net_benefit": [
                    [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
                    [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
                ],
                "strategy_names": ["A", "B", "C"],
                "perspective_names": ["payer", "societal"],
                "perspective_weights": {"payer": 0.25, "societal": 0.75},
                "reference_perspective": "payer",
            }
        ),
        encoding="utf-8",
    )


def test_experimental_perspective_contract_files_are_valid_json() -> None:
    """The experimental contract scaffold should contain parseable JSON."""
    contract_root = Path("specs/frontier/perspective/v1")
    for relative_path in [
        "schemas/perspective-set.schema.json",
        "schemas/value-of-perspective-result.schema.json",
        "examples/value-of-perspective.example.json",
    ]:
        payload = json.loads((contract_root / relative_path).read_text())
        assert isinstance(payload, dict)


def test_perspective_fixture_manifest_and_payload_are_deterministic() -> None:
    """The deterministic fixture set should anchor the CLI contract."""
    fixture_root = Path("specs/frontier/perspective/v1/fixtures")
    manifest = json.loads((fixture_root / "manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert manifest["status"] == "fixture-backed"
    normative = cast("list[dict[str, object]]", manifest["normative"])
    assert len(normative) == 1

    entry = normative[0]
    assert entry["name"] == "screening program perspective comparison"
    assert entry["method_family"] == "value_of_perspective"
    assert entry["input_artifact"] == "normative/perspective-surface.json"
    assert entry["expected_output_artifact"] == "normative/value-of-perspective.json"
    assert entry["tolerance_policy"] == "exact"
    assert entry["provenance"] == {
        "seed": 101,
        "execution_mode": "deterministic",
    }

    surface = fixture_root / "normative" / "perspective-surface.json"
    output = fixture_root / "normative" / "value-of-perspective.json"
    for path in (surface, output):
        assert path.is_file()

    result = runner.invoke(
        cli.app,
        ["--format", "json", "calculate-perspective", str(surface)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    expected = json.loads(output.read_text())
    assert payload == expected


def test_value_of_perspective_compares_conflicting_perspectives() -> None:
    """Different objective functions should expose regret and switching value."""
    net_benefits = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        ]
    )
    perspectives = PerspectiveSet(
        [
            Perspective(id="payer", label="Payer"),
            Perspective(id="societal", label="Societal"),
        ]
    )

    result = value_of_perspective(
        net_benefits,
        perspectives=perspectives,
        strategy_names=["A", "B", "C"],
        perspective_weights={"payer": 0.25, "societal": 0.75},
        reference_perspective="payer",
    )

    assert isinstance(result, ValueOfPerspectiveResult)
    assert result.perspective_ids == ["payer", "societal"]
    assert result.strategy_names == ["A", "B", "C"]
    np.testing.assert_allclose(
        result.expected_net_benefits,
        np.array([[10.0, 8.0, 5.0], [7.0, 11.0, 9.0]]),
    )
    assert result.optimal_strategy_names == ["A", "B"]
    np.testing.assert_allclose(
        result.regret_matrix,
        np.array([[0.0, 2.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(result.switching_values, np.array([0.0, 4.0]))
    assert result.consensus_strategy_name == "B"
    assert result.consensus_weighted_expected_net_benefit == pytest.approx(10.25)
    assert result.robust_strategy_name == "B"
    assert result.pareto_strategy_names == ["A", "B"]
    assert result.method_maturity == "experimental"
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "value_of_perspective"
    assert result.reporting["perspective_ids"] == ["payer", "societal"]


def test_value_of_perspective_accepts_integer_reference_index() -> None:
    """An explicit integer reference perspective should be accepted."""
    result = value_of_perspective(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ]
        ),
        perspective_names=["payer", "societal"],
        reference_perspective=1,
    )

    assert result.reference_perspective_id == "societal"
    assert result.switching_values.tolist() == [2.0, 0.0]
    assert perspective_optimal_strategies(result) == {
        "payer": "Strategy 0",
        "societal": "Strategy 1",
    }


def test_value_of_perspective_identical_perspectives_have_zero_switching_value() -> (
    None
):
    """Identical perspectives should not report artificial switching value."""
    net_benefits = np.array(
        [
            [[2.0, 2.0], [4.0, 4.0]],
            [[3.0, 3.0], [5.0, 5.0]],
        ]
    )

    result = value_of_perspective(net_benefits, perspective_names=["p1", "p2"])

    assert result.optimal_strategy_names == ["Strategy 1", "Strategy 1"]
    assert result.value == pytest.approx(0.0)
    np.testing.assert_allclose(result.regret_matrix, np.zeros((2, 2)))
    np.testing.assert_allclose(result.switching_values, np.zeros(2))


def test_value_of_perspective_accepts_value_array_with_perspective_dimension() -> None:
    """The method should use ValueArray strategy and perspective coordinates."""
    value_array = ValueArray.from_numpy_perspectives(
        np.array(
            [
                [[1.0, 6.0], [3.0, 2.0]],
                [[2.0, 8.0], [4.0, 3.0]],
            ]
        ),
        strategy_names=["usual", "new"],
        perspective_names=["payer", "patient"],
    )

    result = value_of_perspective(value_array)

    assert result.strategy_names == ["usual", "new"]
    assert result.perspective_ids == ["payer", "patient"]
    assert result.optimal_strategy_names == ["new", "usual"]


def test_decision_analysis_wraps_value_of_perspective() -> None:
    """DecisionAnalysis should expose the perspective-comparison method."""
    analysis = DecisionAnalysis(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ]
        )
    )

    result = analysis.value_of_perspective(
        perspective_names=["payer", "societal"],
        strategy_names=["A", "B"],
    )

    assert result.optimal_strategy_names == ["A", "B"]


def test_value_of_perspective_rejects_invalid_inputs() -> None:
    """Invalid dimensions, metadata, and weights should fail early."""
    with pytest.raises(InputError, match="3D"):
        value_of_perspective(np.ones((2, 2)))

    with pytest.raises(InputError, match="perspectives"):
        value_of_perspective(np.ones((2, 2, 2)), perspective_names=["only-one"])

    with pytest.raises(InputError, match="unique"):
        PerspectiveSet([Perspective(id="payer"), Perspective(id="payer")])

    with pytest.raises(InputError, match="weights"):
        value_of_perspective(
            np.ones((2, 2, 2)),
            perspective_names=["payer", "societal"],
            perspective_weights=[1.0],
        )

    with pytest.raises(InputError, match="reference_perspective"):
        value_of_perspective(
            np.ones((2, 2, 2)),
            perspective_names=["payer", "societal"],
            reference_perspective="missing",
        )

    with pytest.raises(InputError, match="ValueArray"):
        value_of_perspective(cast("ValueArray", "not an array"))


def test_value_of_perspective_rejects_additional_validation_failures() -> None:
    """The contract should reject finite, shape, and weight mismatches."""
    with pytest.raises(InputError, match="finite"):
        value_of_perspective(
            np.array([[[np.nan, 1.0], [2.0, 3.0]]]),
            perspective_names=["payer", "societal"],
        )

    with pytest.raises(InputError, match="strategy_names"):
        value_of_perspective(
            np.ones((2, 2, 2)),
            perspective_names=["payer", "societal"],
            strategy_names=["only-one"],
        )

    with pytest.raises(InputError, match="perspective_names"):
        value_of_perspective(
            np.ones((2, 2, 2)),
            perspective_names=["payer"],
        )

    with pytest.raises(InputError, match="weights"):
        value_of_perspective(
            np.ones((2, 2, 2)),
            perspective_names=["payer", "societal"],
            perspective_weights={"payer": 0.0, "societal": 0.0},
        )


def test_perspective_metadata_and_plot_helpers_cover_edge_cases() -> None:
    """Validate perspective metadata and the regret heatmap helper."""
    with pytest.raises(InputError, match="non-empty"):
        Perspective(id="")

    with pytest.raises(InputError, match="finite"):
        Perspective(id="payer", willingness_to_pay=np.inf)

    with pytest.raises(InputError, match="At least one perspective"):
        PerspectiveSet([])

    with pytest.raises(InputError, match="out of range"):
        value_of_perspective(
            np.ones((2, 2, 2)),
            perspective_names=["payer", "societal"],
            reference_perspective=3,
        )

    result = value_of_perspective(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ]
        ),
        perspective_names=["payer", "societal"],
    )
    ax = plot_perspective_regret(result, title="Example Regret")

    assert ax.get_title() == "Example Regret"
    assert len(ax.texts) == 4


def test_cli_calculate_perspective_outputs_json(tmp_path: Path) -> None:
    """The CLI should expose the experimental perspective calculation."""
    surface_file = tmp_path / "perspective.json"
    _write_perspective_surface(surface_file)

    result = runner.invoke(
        cli.app,
        ["--format", "json", "calculate-perspective", str(surface_file)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "calculate-perspective"
    assert payload["method_maturity"] == "experimental"
    assert payload["reporting"]["reporting_standard"] == "CHEERS-VOI"
    assert payload["consensus_strategy"] == "B"
    assert payload["robust_strategy"] == "B"
    assert payload["pareto_strategies"] == ["A", "B"]


def test_cli_plot_perspective_regret_saves_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should expose a perspective regret plot command."""
    surface_file = tmp_path / "perspective.json"
    output_file = tmp_path / "regret.png"
    _write_perspective_surface(surface_file)
    monkeypatch.setattr(cli, "render_perspective_regret", lambda _result: _DummyAxes())

    result = runner.invoke(
        cli.app,
        [
            "--format",
            "json",
            "plot-perspective-regret",
            str(surface_file),
            "-o",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.read_text(encoding="utf-8") == "figure"
    payload = json.loads(result.stdout)
    assert payload["command"] == "plot-perspective-regret"
    assert payload["saved"] is True
