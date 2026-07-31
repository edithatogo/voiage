"""Reference, invariant and pathology tests for issue #559 Value of Flexibility."""

from __future__ import annotations

import json
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


def _stage_weights() -> dict[str, float]:
    return {"now": 0.2, "mid": 0.3, "late": 0.5}


def _provenance() -> dict[str, str]:
    return {"fixture_id": "vof-test-v1", "execution_mode": "deterministic"}


def _calculate(**kwargs: object):
    options = dict(kwargs)
    weights = options.pop("stage_weights", _stage_weights())
    return value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        weights,
        _provenance(),
        **options,
    )


def test_value_of_flexibility_matches_enumerable_commitment_reference() -> None:
    result = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["commit_a", "commit_b"],
        {"now": 0.2, "mid": 0.3, "late": 0.5},
        _provenance(),
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
        _provenance(),
    )
    permuted = value_of_flexibility(
        _surface()[:, ::-1, :],
        ["now", "mid", "late"],
        ["b", "a"],
        {"now": 0.2, "mid": 0.3, "late": 0.5},
        _provenance(),
    )
    assert permuted.value_of_flexibility == pytest.approx(original.value_of_flexibility)
    assert permuted.flexible_value == pytest.approx(original.flexible_value)
    assert permuted.constrained_value == pytest.approx(original.constrained_value)


def test_ties_use_canonical_strategy_names_independent_of_input_order() -> None:
    surface = np.asarray([[[5.0, 5.0], [5.0, 5.0]]])
    weights = {"early": 0.5, "late": 0.5}
    original = value_of_flexibility(
        surface, ["early", "late"], ["z", "a"], weights, _provenance()
    )
    permuted = value_of_flexibility(
        surface[:, ::-1, :],
        ["early", "late"],
        ["a", "z"],
        weights,
        _provenance(),
    )
    assert original.commitment_baseline == permuted.commitment_baseline == "a"
    assert original.flexible_policy_path == permuted.flexible_policy_path == ["a", "a"]
    assert original.diagnostics["commitment_ties"] == ["a", "z"]
    assert original.diagnostics["tie_policy"] == "canonical-lexicographic"


def test_identical_flexible_and_commitment_sets_have_zero_value() -> None:
    result = value_of_flexibility(
        np.asarray([[[3.0, 4.0, 5.0]]]),
        ["now", "mid", "late"],
        ["only"],
        _stage_weights(),
        _provenance(),
    )
    assert result.value_of_flexibility == 0.0
    assert result.flexible_policy_path == ["only"] * 3
    assert result.constrained_policy_path == ["only"] * 3


def test_policy_sets_are_explicit_and_constrained_set_must_be_feasible() -> None:
    result = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        _stage_weights(),
        _provenance(),
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
            _stage_weights(),
            _provenance(),
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
        _calculate(**kwargs)


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
        ({"discount_rate": np.nan}, "must be finite"),
        ({"discount_rate": np.inf}, "must be finite"),
        ({"irreversibility_penalty": np.nan}, "must be finite"),
        ({"lock_in_penalty": np.inf}, "must be finite"),
    ],
)
def test_value_of_flexibility_rejects_invalid_policy_and_numeric_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _calculate(**kwargs)


def test_value_of_flexibility_defends_subset_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "dot", lambda *_args: -100.0)
    with pytest.raises(ValueError, match="below its feasible commitment subset"):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            _stage_weights(),
            _provenance(),
        )


def test_value_of_flexibility_requires_explicit_stage_weights() -> None:
    with pytest.raises(ValueError, match="stage_weights must be declared"):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            None,  # type: ignore[arg-type] - runtime must fail closed too
            _provenance(),
        )


def test_value_of_flexibility_rejects_overflowing_weight_normalization() -> None:
    maximum = np.finfo(float).max
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="finite positive sum"),
    ):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            {"now": maximum, "mid": maximum, "late": maximum},
            _provenance(),
        )


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("discount_rate", 0.01),
        ("irreversibility_penalty", 0.01),
        ("lock_in_penalty", 0.01),
    ],
)
def test_value_of_flexibility_v1_rejects_ungoverned_nonzero_controls(
    control: str, value: float
) -> None:
    with pytest.raises(ValueError, match="unsupported.*v1"):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            _stage_weights(),
            _provenance(),
            **{control: value},
        )


def test_value_of_flexibility_output_preserves_axes_and_provenance() -> None:
    result = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        _stage_weights(),
        _provenance(),
    )

    assert result.decision_stage_names == ["now", "mid", "late"]
    assert result.strategy_names == ["a", "b"]
    assert result.provenance == _provenance()


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        ({"fixture_id": "x"}, "exactly fixture_id and execution_mode"),
        (
            {"fixture_id": "", "execution_mode": "deterministic"},
            "fixture_id must be a non-empty string",
        ),
        (
            {"fixture_id": "x", "execution_mode": "stochastic"},
            "execution_mode must be 'deterministic'",
        ),
    ],
)
def test_value_of_flexibility_rejects_invalid_provenance(
    provenance: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        value_of_flexibility(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            _stage_weights(),
            provenance,
        )


def test_dynamic_real_options_rejects_nonfinite_and_negative_adjustments() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        value_of_dynamic_real_options(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            _stage_weights(),
            discount_rate=np.nan,
        )

    with pytest.raises(ValueError, match="imply negative value"):
        value_of_dynamic_real_options(
            _surface(),
            ["now", "mid", "late"],
            ["a", "b"],
            _stage_weights(),
            irreversibility_penalty=2.0,
            evidence_arrival_times={"now": 0.0, "mid": 1.0, "late": 2.0},
        )


def test_value_of_flexibility_reports_ordered_scenario_policy_changes() -> None:
    result = value_of_flexibility(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        _stage_weights(),
        _provenance(),
    )

    assert result.exercise_decisions is None
    assert result.ordered_scenario_policy_changes == [False, True, False]


def test_value_of_flexibility_rejects_nonfinite_post_arithmetic_results() -> None:
    maximum = np.finfo(float).max
    overflowing_surface = np.full((2, 2, 3), maximum)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        value_of_flexibility(
            overflowing_surface,
            ["now", "mid", "late"],
            ["a", "b"],
            _stage_weights(),
            _provenance(),
        )


def test_value_of_flexibility_cli_returns_versioned_json(tmp_path: Path) -> None:
    fixture = "specs/frontier/value-of-flexibility/v1/fixtures/normative/input.json"
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
    payload = json.loads(result.stdout)
    assert payload["analysis_type"] == "value_of_flexibility"
    assert payload["value_of_flexibility"] == pytest.approx(1.6)
    assert payload["information_value_component"] == 0.0
    assert payload["decision_stage_names"] == ["now", "mid", "late"]
    assert payload["strategy_names"] == ["commit_a", "commit_b"]
    assert payload["provenance"] == {
        "fixture_id": "vof-enumerable-v1",
        "execution_mode": "deterministic",
    }
    assert payload["exercise_decisions"] is None
    assert payload["ordered_scenario_policy_changes"] == [False, True, False]
    assert '"value_of_flexibility"' in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "{}",
        "{",
        json.dumps(
            {
                "decision_stage_names": ["now"],
                "strategy_names": ["a"],
                "net_benefit": [[[1.0]]],
                "stage_semantics": "timing_scenarios",
                "information_value_included": False,
                "provenance": {
                    "fixture_id": "missing-unit",
                    "execution_mode": "deterministic",
                },
            }
        ),
        json.dumps(
            {
                "decision_stage_names": ["now"],
                "strategy_names": ["a"],
                "net_benefit": [[[1.0]]],
                "value_unit": "point",
                "stage_semantics": "timing_scenarios",
                "information_value_included": False,
                "provenance": {
                    "fixture_id": "unknown-field",
                    "execution_mode": "deterministic",
                },
                "unknown": True,
            }
        ),
    ],
)
def test_value_of_flexibility_cli_rejects_invalid_requests(
    tmp_path: Path, content: str
) -> None:
    request = tmp_path / "invalid.json"
    request.write_text(content, encoding="utf-8")
    result = CliRunner().invoke(app, ["calculate-value-of-flexibility", str(request)])
    assert result.exit_code == 1
    assert "Error:" in result.stderr


def test_legacy_dynamic_real_options_preserves_zero_time_default() -> None:
    omitted = value_of_dynamic_real_options(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        discount_rate=0.2,
    )
    explicit = value_of_dynamic_real_options(
        _surface(),
        ["now", "mid", "late"],
        ["a", "b"],
        discount_rate=0.2,
        evidence_arrival_times={"now": 0.0, "mid": 0.0, "late": 0.0},
    )
    assert omitted.option_value == pytest.approx(explicit.option_value)
    assert omitted.diagnostics["evidence_arrival_times"] == {
        "now": 0.0,
        "mid": 0.0,
        "late": 0.0,
    }
