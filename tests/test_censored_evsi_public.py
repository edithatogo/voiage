"""Exercise public censored EVSI on named paired-draw scopes."""

from __future__ import annotations

from fractions import Fraction
from importlib import import_module
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pytest

from voiage.contracts.normalized_input import VOIBinding
from voiage.exceptions import DimensionMismatchError
from voiage.ingestion.dataframe import from_dataframe

if TYPE_CHECKING:
    from voiage.contracts.normalized_input import NormalizedInputBundle
    from voiage.experimental.censored_evsi import CensoredSequentialEvsiResult

_STRATEGIES = ("keep", "switch")


def _data() -> pa.Table:
    return pa.table(
        {
            "draw_id": [9, 7, 9, 7, 9, 9],
            "strategy_id": ["switch", "keep", "keep", "switch", "keep", "switch"],
            "perspective": ["payer", "payer", "payer", "payer", "societal", "societal"],
            "subgroup_id": ["trial", "trial", "trial", "trial", "other", "other"],
            "net_benefit": [10, 6, 0, 0, 1_000, 0],
            "draw_weight": [1, 1, 1, 1, 100, 100],
        }
    )


def _calculate(
    *,
    table: pa.Table | None = None,
    bundle: NormalizedInputBundle | None = None,
    weight_field: str | None = "draw_weight",
    second_draw: object = 9,
    perspective: str = "payer",
    budget: int = 128,
) -> CensoredSequentialEvsiResult:
    experimental = import_module("voiage.experimental")
    evaluate = experimental.censored_sequential_evsi
    if bundle is None:
        bundle = experimental.paired_draw_bundle(
            _data() if table is None else table,
            dataset_id="censored-test",
            strategy_ids=_STRATEGIES,
            weight_field=weight_field,
        )
    acquisitions = [
        [
            {
                "action_id": "inspection",
                "limit": 0.0,
                "cutpoints": (),
                "outcome_ids": ("unavailable", "below_limit", "reading"),
                "states": (
                    {
                        "draw_id": 7,
                        "mean": 0.0,
                        "standard_deviation": 1.0,
                        "detection_probability": 0.0,
                    },
                    {
                        "draw_id": second_draw,
                        "mean": 0.0,
                        "standard_deviation": 1.0,
                        "detection_probability": 0.5,
                    },
                ),
            }
        ]
    ]
    return cast(
        "CensoredSequentialEvsiResult",
        evaluate(
            bundle,
            stages=acquisitions,
            perspective=perspective,
            subgroup_id="trial",
            max_branches=budget,
        ),
    )


def test_censored_evsi_preserves_draw_keys_scope_and_exact_values() -> None:
    result = _calculate()
    assert result.status == "complete"
    assert (result.perspective, result.subgroup_id) == ("payer", "trial")
    assert result.draw_ids == (9, 7)
    assert result.strategy_ids == _STRATEGIES
    assert result.current_value == Fraction(5)
    assert result.expected_value == Fraction(11, 2)
    assert result.evsi == Fraction(1, 2)


def test_censored_evsi_reports_budget_and_key_mismatch() -> None:
    result = _calculate(budget=1)
    assert result.status == "indeterminate"
    assert result.current_value == Fraction(5)
    assert result.evsi is None
    assert result.expected_value is None
    assert result.branches_examined == 1
    with pytest.raises(DimensionMismatchError, match="state IDs"):
        _calculate(second_draw=12)


def test_censored_evsi_requires_a_declared_weight() -> None:
    with pytest.raises(ValueError, match="weight binding"):
        _calculate(weight_field=None)


def test_censored_evsi_rejects_lossy_integer_value_conversion() -> None:
    table = _data().set_column(
        4, "net_benefit", pa.array([2**53 + 1, 6, 0, 0, 1_000, 0])
    )
    with pytest.raises(ValueError, match="exactly representable"):
        _calculate(table=table)


@pytest.mark.parametrize("budget", [0, True])
def test_censored_evsi_rejects_invalid_branch_budgets(budget: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _calculate(budget=budget)


def test_censored_evsi_rejects_missing_scope_and_malformed_observation_id() -> None:
    with pytest.raises(ValueError, match="No paired draws"):
        _calculate(perspective="unlisted")
    with pytest.raises(ValueError, match="integer or text draw ID"):
        _calculate(second_draw=1.25)


def test_censored_evsi_rejects_wide_payoff_binding() -> None:
    bundle = from_dataframe(
        _data(),
        dataset_id="wide-censored-test",
        bindings=(
            VOIBinding(role="net_benefit", table_id="data", field_ids=("net_benefit",)),
            VOIBinding(role="weight", table_id="data", field_ids=("draw_weight",)),
        ),
        allow_copy=False,
    )
    with pytest.raises(ValueError, match="aligned paired-draw"):
        _calculate(bundle=bundle)


def test_censored_evsi_rejects_incompatible_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("voiage.experimental.censored_evsi")
    compute = module.compute_censored_sequential_evsi

    def mismatched_states(experiment: str, *, max_branches: int) -> dict[str, object]:
        result = compute(experiment, max_branches=max_branches)
        result["state_ids"] = ["7", "9"]
        return result

    monkeypatch.setattr(module, "compute_censored_sequential_evsi", mismatched_states)
    with pytest.raises(RuntimeError, match="incompatible contract"):
        _calculate()
