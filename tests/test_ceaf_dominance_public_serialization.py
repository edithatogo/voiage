"""Stable public CEAF and dominance serialization contracts."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest
import xarray as xr

import voiage
from voiage.exceptions import InputError

REPO_ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict[str, object]:
    path = REPO_ROOT / f"specs/core-api/schemas/v1/results/{name}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ceaf_result() -> voiage.CEAFResult:
    values = np.array(
        [
            [[5.0, 1.0], [0.0, 2.0]],
            [[5.0, 1.0], [0.0, 2.0]],
        ]
    )
    value_array = voiage.ValueArray(
        dataset=xr.Dataset(
            {
                "net_benefit": (
                    ("n_samples", "n_strategies", "n_wtp"),
                    values,
                )
            },
            coords={"strategy": ("n_strategies", ["A", "B"])},
        )
    )
    return voiage.ceaf(value_array, [0.0, 1.0])


def test_stable_entry_points_and_result_types_are_public() -> None:
    assert callable(voiage.ceaf)
    assert callable(voiage.dominance)
    assert "ceaf" in voiage.__all__
    assert "dominance" in voiage.__all__
    assert "CEAFResult" in voiage.__all__
    assert "DominanceResult" in voiage.__all__


def test_ceaf_runtime_result_serializes_to_v1_schema() -> None:
    payload = _ceaf_result().to_dict(
        analysis_id="ceaf-001",
        decision_problem_id="decision-001",
    )

    Draft202012Validator(_schema("ceaf")).validate(payload)
    json.dumps(payload, allow_nan=False)
    lengths = {
        len(payload[field])
        for field in (
            "wtp_thresholds",
            "optimal_strategy_indices",
            "optimal_strategy_names",
            "acceptability_probabilities",
            "probability_lower",
            "probability_upper",
            "expected_net_benefit",
        )
    }
    assert lengths == {2}


def test_dominance_runtime_result_serializes_to_v1_schema() -> None:
    payload = voiage.dominance(
        costs=[100.0, 200.0, 500.0, 800.0, 900.0],
        effects=[1.0, 2.0, 2.5, 4.0, 3.5],
        strategy_names=["A", "B", "C", "D", "E"],
    ).to_dict(
        analysis_id="dominance-001",
        decision_problem_id="decision-001",
    )

    Draft202012Validator(_schema("dominance")).validate(payload)
    json.dumps(payload, allow_nan=False)
    assert len(payload["strategy_names"]) == len(payload["costs"])
    assert len(payload["costs"]) == len(payload["effects"])
    assert len(payload["effects"]) == len(payload["status"])
    frontier_steps = len(payload["frontier_indices"]) - 1
    assert len(payload["incremental_costs"]) == frontier_steps
    assert len(payload["incremental_effects"]) == frontier_steps
    assert len(payload["icers"]) == frontier_steps


def test_serialization_requires_non_empty_identifiers() -> None:
    result = _ceaf_result()
    with pytest.raises(InputError, match="analysis_id"):
        result.to_dict(analysis_id="", decision_problem_id="decision-001")
    with pytest.raises(InputError, match="decision_problem_id"):
        result.to_dict(analysis_id="ceaf-001", decision_problem_id="")
