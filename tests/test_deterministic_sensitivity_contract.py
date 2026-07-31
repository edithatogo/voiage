"""Versioned wire-contract tests for deterministic sensitivity analysis."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/deterministic-sensitivity-analysis/v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dsa_normative_input_and_result_validate() -> None:
    input_schema = _json(
        CONTRACT / "schemas/deterministic-sensitivity-input.schema.json"
    )
    result_schema = _json(
        CONTRACT / "schemas/deterministic-sensitivity-result.schema.json"
    )
    Draft202012Validator(input_schema).validate(
        _json(CONTRACT / "fixtures/normative/input.json")
    )
    Draft202012Validator(result_schema).validate(
        _json(CONTRACT / "fixtures/normative/expected.json")
    )


def test_dsa_schemas_reject_unknown_and_incomplete_payloads() -> None:
    schema = _json(CONTRACT / "schemas/deterministic-sensitivity-input.schema.json")
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload["unknown"] = True
    with pytest.raises(Exception, match="Additional properties"):
        Draft202012Validator(schema).validate(payload)
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    payload.pop("output_unit")
    with pytest.raises(Exception, match="output_unit"):
        Draft202012Validator(schema).validate(payload)


def test_dsa_capabilities_remain_fail_closed_before_runtime() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "planned"
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "planned"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"
    assert capabilities["stable_claim_allowed"] is False


def test_dsa_fixture_evidence_is_exact_and_hash_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    expected = {
        "specs/frontier/deterministic-sensitivity-analysis/v1/schemas/deterministic-sensitivity-input.schema.json",
        "specs/frontier/deterministic-sensitivity-analysis/v1/schemas/deterministic-sensitivity-result.schema.json",
        "specs/frontier/deterministic-sensitivity-analysis/v1/capabilities.json",
        "specs/frontier/deterministic-sensitivity-analysis/v1/fixtures/manifest.json",
        "specs/frontier/deterministic-sensitivity-analysis/v1/fixtures/normative/input.json",
        "specs/frontier/deterministic-sensitivity-analysis/v1/fixtures/normative/expected.json",
        "conductor/tracks/supported_frontier_method_completion_20260723/deterministic-sensitivity-reference-review.md",
    }
    assert {item["path"] for item in evidence["artifacts"]} == expected
    for item in evidence["artifacts"]:
        assert (
            hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def test_dsa_runtime_fixture_conformance_is_required_before_execution_claim() -> None:
    module = importlib.import_module("voiage.methods.deterministic_sensitivity")
    result = module.deterministic_sensitivity_from_specification(
        _json(CONTRACT / "fixtures/normative/input.json")
    )
    assert result.to_contract_dict() == _json(
        CONTRACT / "fixtures/normative/expected.json"
    )
