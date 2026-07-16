"""Contract tests for the monitoring-surveillance frontier family."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_monitoring_surveillance_fixture_contract_validates() -> None:
    root = Path(__file__).parents[1] / "specs/frontier/monitoring-surveillance/v1"
    input_schema = json.loads(
        (root / "schemas/monitoring-surveillance-set.schema.json").read_text()
    )
    output_schema = json.loads(
        (
            root / "schemas/value-of-monitoring-surveillance-result.schema.json"
        ).read_text()
    )
    fixture = json.loads(
        (root / "fixtures/normative/monitoring-surveillance-set.json").read_text()
    )
    result = json.loads(
        (root / "fixtures/normative/value-of-monitoring-surveillance.json").read_text()
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
