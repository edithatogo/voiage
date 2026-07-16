"""Contract tests for implementation-strategy comparison fixtures."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_implementation_strategy_fixture_contract_validates() -> None:
    root = Path(__file__).parents[1] / "specs/frontier/implementation-strategy/v1"
    input_schema = json.loads(
        (root / "schemas/implementation-strategy-set.schema.json").read_text()
    )
    output_schema = json.loads(
        (
            root / "schemas/value-of-implementation-strategy-result.schema.json"
        ).read_text()
    )
    fixture = json.loads(
        (root / "fixtures/normative/implementation-strategy-set.json").read_text()
    )
    result = json.loads(
        (root / "fixtures/normative/value-of-implementation-strategy.json").read_text()
    )
    Draft202012Validator(input_schema).validate(
        {
            k: v
            for k, v in fixture.items()
            if k not in {"net_benefit", "analysis_id", "decision_problem_id"}
        }
    )
    Draft202012Validator(output_schema).validate(result)
    assert result["method_maturity"] == "fixture-backed"
