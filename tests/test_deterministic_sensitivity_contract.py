"""Versioned wire-contract tests for deterministic sensitivity analysis."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator
import pytest

from voiage.deterministic_sensitivity_contract import (
    validate_deterministic_sensitivity_specification,
)
from voiage.exceptions import InputError

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


def test_dsa_capabilities_are_experimental_python_only() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "executable"
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
        "conductor/archive/supported_frontier_method_completion_20260723/deterministic-sensitivity-reference-review.md",
        "voiage/deterministic_sensitivity_contract.py",
        "voiage/cli.py",
        "voiage/methods/deterministic_sensitivity.py",
        "voiage/plot/deterministic_sensitivity.py",
        "tests/test_deterministic_sensitivity.py",
        "tests/test_deterministic_sensitivity_contract.py",
        "tests/test_deterministic_sensitivity_surfaces.py",
        "docs/astro-site/src/content/docs/examples/deterministic-sensitivity-analysis.mdx",
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


def test_dsa_runtime_contract_rejects_cross_field_inconsistencies() -> None:
    invalid_cases: list[tuple[dict[str, object], str]] = []

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    grids = cast("list[dict[str, object]]", payload["parameter_grids"])
    grids[1]["parameter_name"] = "z"
    invalid_cases.append((payload, "must name the same parameters"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    grids = cast("list[dict[str, object]]", payload["parameter_grids"])
    grids[1]["unit"] = "different-unit"
    invalid_cases.append((payload, "units must match exactly"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    designs = cast("list[dict[str, object]]", payload["two_way_designs"])
    designs[0]["second_parameter"] = "x"
    designs[0]["surface_id"] = "x|x"
    invalid_cases.append((payload, "two distinct baseline parameters"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    designs = cast("list[dict[str, object]]", payload["two_way_designs"])
    designs[0]["surface_id"] = "wrong-surface"
    invalid_cases.append((payload, "surface_id must be"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    designs = cast("list[dict[str, object]]", payload["two_way_designs"])
    points = cast("list[dict[str, object]]", designs[0]["feasible_points"])
    points.append(points[0].copy())
    invalid_cases.append((payload, "feasible_points must be unique"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    designs = cast("list[dict[str, object]]", payload["two_way_designs"])
    designs[0]["feasibility_semantics"] = "full-cartesian-independent"
    invalid_cases.append((payload, "requires the exact Cartesian grid"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    records = cast("list[dict[str, object]]", payload["model_evaluation_records"])
    coordinates = cast("list[dict[str, object]]", records[0]["coordinates"])
    coordinates.pop()
    invalid_cases.append((payload, "complete baseline coordinate set"))

    payload = _json(CONTRACT / "fixtures/normative/input.json")
    records = cast("list[dict[str, object]]", payload["model_evaluation_records"])
    outputs = cast("list[dict[str, object]]", records[0]["alternative_outputs"])
    outputs.pop()
    invalid_cases.append((payload, "exactly alternative_names"))

    for invalid_payload, message in invalid_cases:
        with pytest.raises(InputError, match=message):
            validate_deterministic_sensitivity_specification(invalid_payload)


def test_dsa_runtime_contract_accepts_exact_cartesian_design_before_next_design() -> (
    None
):
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    designs = cast("list[dict[str, object]]", payload["two_way_designs"])
    designs[0]["feasibility_semantics"] = "full-cartesian-independent"
    designs[0]["feasible_points"] = [
        {"first": first, "second": second}
        for first in (-2.0, 0.0, 2.0)
        for second in (-1.0, 0.0, 1.0)
    ]
    designs.append(
        {
            "surface_id": "y|x",
            "first_parameter": "y",
            "second_parameter": "x",
            "feasibility_semantics": "explicit-mask",
            "feasible_points": [{"first": 0.0, "second": 0.0}],
        }
    )

    validate_deterministic_sensitivity_specification(payload)
