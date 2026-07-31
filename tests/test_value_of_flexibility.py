"""Reference, invariant and pathology tests for issue #559 Value of Flexibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from voiage.cli import app
from voiage.methods.dynamic_real_options import (
    value_of_dynamic_real_options,
    value_of_flexibility,
)

if TYPE_CHECKING:
    from pathlib import Path


def _surface() -> np.ndarray:
    return np.asarray([[[8.0, 8.0, 8.0], [0.0, 10.0, 12.0]]])


def test_value_of_flexibility_matches_enumerable_commitment_reference() -> None:
    result = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["commit_a", "commit_b"],
        {"now": 0.2, "mid": 0.3, "late": 0.5},
        value_unit="net-benefit-point",
    )

    assert result.flexible_value == pytest.approx(10.6)
    assert result.constrained_value == pytest.approx(9.0)
    assert result.value_of_flexibility == pytest.approx(1.6)
    assert result.flexible_policy_path == ["commit_a", "commit_b", "commit_b"]
    assert result.constrained_policy_path == ["commit_b"] * 3
    assert result.commitment_baseline == "commit_b"
    assert result.value_unit == "net-benefit-point"
    assert result.information_value_component == 0.0


def test_value_of_flexibility_is_invariant_to_strategy_permutation() -> None:
    original = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        {"now": 0.2, "mid": 0.3, "late": 0.5},
    )
    permuted = value_of_flexibility(
        _surface()[:, ::-1, :],
        ["now", "mid", "late"],
        ["b", "a"],
        {"now": 0.2, "mid": 0.3, "late": 0.5},
    )
    assert permuted.value_of_flexibility == pytest.approx(
        original.value_of_flexibility
    )
    assert permuted.flexible_value == pytest.approx(original.flexible_value)
    assert permuted.constrained_value == pytest.approx(original.constrained_value)


def test_identical_flexible_and_commitment_sets_have_zero_value() -> None:
    result = value_of_flexibility(
        np.asarray([[[3.0, 4.0, 5.0]]]),
        ["now", "mid", "late"],
        ["only"],
    )
    assert result.value_of_flexibility == 0.0
    assert result.flexible_policy_path == ["only"] * 3
    assert result.constrained_policy_path == ["only"] * 3


def test_policy_sets_are_explicit_and_constrained_set_must_be_feasible() -> None:
    result = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        flexible_policy_sets={
            "now": ["a"],
            "mid": ["a", "b"],
            "late": ["a", "b"],
        },
        constrained_strategy_names=["a"],
    )
    assert result.commitment_baseline == "a"
    assert result.value_of_flexibility >= 0.0

    with pytest.raises(ValueError, match="feasible in every timing scenario"):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            flexible_policy_sets={
                "now": ["a"],
                "mid": ["a", "b"],
                "late": ["a", "b"],
            },
            constrained_strategy_names=["b"],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stage_weights": {"now": 1.0}}, "exactly match"),
        (
            {"evidence_arrival_times": {"now": 0.0, "mid": 2.0, "late": 1.0}},
            "strictly increasing",
        ),
        ({"value_unit": ""}, "value_unit"),
        ({"information_value_included": True}, "double counting"),
        ({"stage_semantics": "lifecycle_periods"}, "timing_scenarios"),
    ],
)
def test_value_of_flexibility_fails_closed_on_ambiguous_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            **kwargs,
        )


def test_dynamic_real_options_compatibility_uses_stagewise_commitment_math() -> None:
    result = value_of_dynamic_real_options(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        {"now": 0.2, "mid": 0.3, "late": 0.5},
    )
    assert result.option_value == pytest.approx(1.6)
    assert result.diagnostics["commitment_baseline"] == "b"
    assert result.reporting["adjacent_estimand"] == "value_of_flexibility"

    with pytest.raises(ValueError, match="exercise_rules are not executable"):
        value_of_dynamic_real_options(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            exercise_rules={"now": "exercise"},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"flexible_policy_sets": {"now": ["a"], "mid": ["a"]}},
            "exactly match",
        ),
        (
            {
                "flexible_policy_sets": {
                    "now": [],
                    "mid": ["a"],
                    "late": ["a"],
                }
            },
            "non-empty and unique",
        ),
        (
            {
                "flexible_policy_sets": {
                    "now": ["missing"],
                    "mid": ["a"],
                    "late": ["a"],
                }
            },
            "Unknown flexible strategies",
        ),
        ({"constrained_strategy_names": []}, "non-empty and unique"),
        ({"constrained_strategy_names": ["missing"]}, "Unknown constrained"),
        (
            {"stage_weights": {"now": 1.0, "mid": np.nan, "late": 1.0}},
            "finite values",
        ),
        (
            {
                "irreversibility_penalty": 2.0,
                "evidence_arrival_times": {"now": 0.0, "mid": 1.0, "late": 2.0},
            },
            "negative value",
        ),
    ],
)
def test_value_of_flexibility_rejects_invalid_policy_and_numeric_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            **kwargs,
        )


def test_value_of_flexibility_defends_subset_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "dot", lambda *_args: -100.0)
    with pytest.raises(ValueError, match="below its feasible commitment subset"):
        value_of_flexibility(
            _surface(), ["now", "mid", "late"], ["a", "b"]
        )


def test_value_of_flexibility_cli_returns_versioned_json(tmp_path: Path) -> None:
    fixture = (
        "specs/frontier/value-of-flexibility/v1/fixtures/normative/input.json"
    )
    output = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-value-of-flexibility",
            fixture,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert '"analysis_type": "value_of_flexibility"' in result.stdout
    assert '"value_of_flexibility": 1.5999999999999996' in result.stdout
    assert '"information_value_component": 0.0' in result.stdout
    assert '"value_of_flexibility"' in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("content", ["[]", "{}", "{"])
def test_value_of_flexibility_cli_rejects_invalid_requests(
    tmp_path: Path, content: str
) -> None:
    request = tmp_path / "invalid.json"
    request.write_text(content, encoding="utf-8")
    result = CliRunner().invoke(
        app, ["calculate-value-of-flexibility", str(request)]
    )
    assert result.exit_code == 1
    assert "Error:" in result.stderr
