"""Portable contract tests for issue #558 qualitative VOI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.qualitative_information import (
    QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1,
    validate_qualitative_information_semantics,
)
from voiage.methods.qualitative_information import (
    qualitative_information_from_specification,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/qualitative-information/v1"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_normative_fixture_and_portable_schemas_validate() -> None:
    names = (
        "qualitative-information-assessment.schema.json",
        "qualitative-information-result.schema.json",
        "qualitative-information-audit-event.schema.json",
        "qualitative-information-rendering.schema.json",
    )
    schemas = [_json(CONTRACT / "schemas" / name) for name in names]
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    rendering = _json(CONTRACT / "fixtures/normative/rendering.json")
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(schemas[0]).validate(payload)
    Draft202012Validator(schemas[1]).validate(expected)
    Draft202012Validator(schemas[3]).validate(rendering)
    for event in payload["audit_history"]:
        Draft202012Validator(schemas[2]).validate(event)
    validate_qualitative_information_semantics(payload)
    assert schemas[0] == QUALITATIVE_INFORMATION_ASSESSMENT_SCHEMA_V1


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("decision", "alternatives"), ["same", "same"], "unique"),
        (("assessment_version",), 0, "positive"),
        (("audit_history", 1, "previous_event_id"), "wrong", "chain"),
        (("questions", 0, "judgements", 0, "priority_class"), 3, "string"),
    ],
)
def test_contract_and_semantics_fail_closed(
    path: tuple[object, ...], value: object, message: str
) -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    target: object = payload
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(Exception, match=message):
        validate_qualitative_information_semantics(payload)


def test_capabilities_do_not_overclaim_bindings_or_maturity() -> None:
    capabilities = _json(CONTRACT / "capabilities.json")
    assert capabilities["maturity"] == "experimental"
    assert capabilities["numerical_estimand"] is False
    surfaces = capabilities["surfaces"]
    assert surfaces["python"]["status"] == "executable"
    assert surfaces["rust"]["status"] == "unsupported"
    assert surfaces["r"]["status"] == "unsupported"
    assert surfaces["julia"]["status"] == "unsupported"
    assert surfaces["mojo"]["status"] == "external"


def test_contract_evidence_is_sha256_pinned() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_runtime_matches_normative_fixture() -> None:
    payload = _json(CONTRACT / "fixtures/normative/input.json")
    expected = _json(CONTRACT / "fixtures/normative/expected.json")
    assert (
        qualitative_information_from_specification(payload).to_contract_dict()
        == expected
    )
