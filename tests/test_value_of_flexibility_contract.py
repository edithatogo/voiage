"""Schema, fixture and capability contracts for issue #559."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np

from voiage.methods import dynamic_real_options

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/value-of-flexibility/v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_value_of_flexibility_normative_fixture_validates() -> None:
    input_schema = _json(CONTRACT / "schemas/value-of-flexibility-input.schema.json")
    result_schema = _json(CONTRACT / "schemas/value-of-flexibility-result.schema.json")
    input_payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    Draft202012Validator(input_schema).validate(input_payload)
    Draft202012Validator(result_schema).validate(expected)


def test_capabilities_fail_closed_for_unimplemented_bindings() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "planned-runtime"
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "planned"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"


def test_manifest_is_single_versioned_normative_reference() -> None:
    manifest = _json(CONTRACT / "fixtures/manifest.json")
    assert manifest["method_family"] == "value_of_flexibility"
    assert manifest["status"] == "experimental"
    assert manifest["fixtures"] == [
        {
            "fixture_id": "vof-enumerable-v1",
            "input": "normative/input.json",
            "expected": "normative/expected.json",
            "reference": "conductor/tracks/supported_frontier_method_completion_20260723/value-of-flexibility-reference-review.md",
            "tolerance": {"absolute": 1e-12, "relative": 1e-12},
        }
    ]


def test_runtime_matches_normative_fixture() -> None:
    input_payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    result = dynamic_real_options.value_of_flexibility(
        np.asarray(input_payload["net_benefit"], dtype=float),
        input_payload["decision_stage_names"],
        input_payload["strategy_names"],
        input_payload["stage_weights"],
        discount_rate=input_payload["discount_rate"],
        irreversibility_penalty=input_payload["irreversibility_penalty"],
        lock_in_penalty=input_payload["lock_in_penalty"],
        evidence_arrival_times=input_payload["evidence_arrival_times"],
        flexible_policy_sets=input_payload["flexible_policy_sets"],
        constrained_strategy_names=input_payload["constrained_strategy_names"],
        value_unit=input_payload["value_unit"],
        stage_semantics=input_payload["stage_semantics"],
        information_value_included=input_payload["information_value_included"],
    )
    actual = asdict(result)
    actual["policy_path_regret"] = result.policy_path_regret.tolist()
    assert actual == expected
