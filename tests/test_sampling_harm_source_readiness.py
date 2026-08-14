"""Fail-closed tests for H8 source and remediation preparation."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from voiage.sampling_harm_source_readiness import (
    SamplingHarmSourceReadinessError,
    _load_object,
    validate_repository_sampling_harm_source_readiness,
    validate_sampling_harm_source_readiness,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/sampling-acquisition-harm/v1"
SCHEMAS = CONTRACT / "schemas"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _artifacts() -> tuple[dict[str, Any], ...]:
    return (
        _json(CONTRACT / "source-observation-refresh-20260803.json"),
        _json(CONTRACT / "remediation-readiness-delta-20260803.json"),
        _json(CONTRACT / "remediation-register.json"),
        _json(SCHEMAS / "source-observation-refresh.schema.json"),
        _json(SCHEMAS / "remediation-readiness-delta.schema.json"),
    )


def _validate(
    source: dict[str, Any],
    delta: dict[str, Any],
    register: dict[str, Any],
    source_schema: dict[str, Any],
    delta_schema: dict[str, Any],
) -> None:
    validate_sampling_harm_source_readiness(
        source,
        delta,
        register,
        source_schema=source_schema,
        delta_schema=delta_schema,
    )


def _validate_semantics(
    source: dict[str, Any],
    delta: dict[str, Any],
    register: dict[str, Any],
) -> None:
    permissive_schema = {"type": "object"}
    validate_sampling_harm_source_readiness(
        source,
        delta,
        register,
        source_schema=permissive_schema,
        delta_schema=permissive_schema,
    )


def test_canonical_source_readiness_is_fail_closed() -> None:
    assert validate_repository_sampling_harm_source_readiness(ROOT) == {
        "sources": 6,
        "findings": 19,
        "pending": 19,
        "replacement_packet": "not_ready_to_freeze",
    }


def test_loader_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SamplingHarmSourceReadinessError, match="contain an object"):
        _load_object(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source, _delta, _register: source["authority_boundary"].__setitem__(
                "scientific_authority", True
            ),
            "source observation invalid",
        ),
        (
            lambda source, _delta, _register: source["sources"][0].__setitem__(
                "retained_source_bytes", True
            ),
            "source observation invalid",
        ),
        (
            lambda source, _delta, _register: source["sources"][1].__setitem__(
                "current_observation_sha256", "f" * 64
            ),
            "Belmont observation must not claim a digest",
        ),
        (
            lambda _source, delta, _register: delta["groups"][
                "candidate_and_human_review_prerequisite"
            ].__setitem__(0, "H8D-API-GOV-01"),
            "finding readiness groups overlap",
        ),
        (
            lambda _source, _delta, register: register["findings"][0].__setitem__(
                "disposition_status", "resolved"
            ),
            "finding is not pending",
        ),
        (
            lambda _source, delta, _register: delta["authority_boundary"].__setitem__(
                "h8_d_satisfied", True
            ),
            "remediation delta invalid",
        ),
        (
            lambda _source, delta, _register: delta["governance_readback"][
                "project"
            ].__setitem__("sync_state", "Conflict"),
            "remediation delta invalid",
        ),
    ],
)
def test_source_readiness_mutations_fail_closed(mutation: Any, message: str) -> None:
    source, delta, register, source_schema, delta_schema = map(deepcopy, _artifacts())
    mutation(source, delta, register)
    with pytest.raises(SamplingHarmSourceReadinessError, match=message):
        _validate(source, delta, register, source_schema, delta_schema)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda source, _delta, _register: source["sources"][0].__setitem__(
                "stable_source_id", source["sources"][1]["stable_source_id"]
            ),
            "source inventory contains duplicates",
        ),
        (
            lambda source, _delta, _register: source["sources"][0].__setitem__(
                "stable_source_id", "unexpected-source"
            ),
            "source inventory mismatch",
        ),
        (
            lambda source, _delta, _register: source["sources"][3].__setitem__(
                "drift_assessment", "unexpected"
            ),
            "byte-stable source set mismatch",
        ),
        (
            lambda source, _delta, _register: source["sources"][3].__setitem__(
                "current_observation_sha256", "f" * 64
            ),
            "byte-stable digest mismatch",
        ),
        (
            lambda source, _delta, _register: source["sources"][2].__setitem__(
                "representation", "text/html"
            ),
            "eCFR representation change must remain non-comparable",
        ),
        (
            lambda _source, _delta, register: register.__setitem__("findings", None),
            "remediation findings are absent",
        ),
        (
            lambda _source, _delta, register: register["findings"].pop(),
            "must retain nineteen findings",
        ),
        (
            lambda _source, delta, _register: delta["groups"][
                "repository_implemented_awaiting_independent_rereview"
            ].__setitem__(0, "H8D-ED-01"),
            "repository finding set mismatch",
        ),
        (
            lambda _source, delta, _register: delta["groups"][
                "source_review_prerequisite"
            ].__setitem__(0, "H8D-ED-01"),
            "source finding set mismatch",
        ),
        (
            lambda _source, delta, _register: delta["groups"][
                "candidate_and_human_review_prerequisite"
            ].__setitem__(0, 7),
            "contains a non-string",
        ),
        (
            lambda _source, delta, _register: delta["groups"][
                "candidate_and_human_review_prerequisite"
            ].__setitem__(0, "unknown-finding"),
            "do not partition the register",
        ),
        (
            lambda _source, _delta, register: next(
                item
                for item in register["findings"]
                if item["finding_id"] == "H8D-DS-03"
            ).__setitem__("severity", "High"),
            "Critical H8D-DS-03 is not preserved",
        ),
    ],
)
def test_semantic_guards_reject_schema_validity_bypasses(
    mutation: Any, message: str
) -> None:
    source, delta, register, _source_schema, _delta_schema = map(deepcopy, _artifacts())
    mutation(source, delta, register)
    with pytest.raises(SamplingHarmSourceReadinessError, match=message):
        _validate_semantics(source, delta, register)
