"""Fail-closed contract tests for v1.1 stable-core promotion evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

EVIDENCE_PATH = Path("specs/v1/stable-core-promotion-evidence.json")
SCHEMA_PATH = Path("specs/v1/stable-core-promotion-evidence.schema.json")
STATUS_PATH = Path("specs/v1/stable-core-status.json")
BINDING_PATH = Path("specs/v1/binding-matrix.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_promotion_evidence_validates_and_binds_current_status() -> None:
    evidence = _load(EVIDENCE_PATH)
    Draft202012Validator(_load(SCHEMA_PATH)).validate(evidence)

    digest = hashlib.sha256(STATUS_PATH.read_bytes()).hexdigest()
    assert evidence["stable_core_status"]["sha256"] == digest
    assert evidence["stable_core_status"]["path"] == str(STATUS_PATH)


def test_promotion_evidence_fails_closed_on_open_gates() -> None:
    evidence = _load(EVIDENCE_PATH)

    assert evidence["promotion"]["release_target"] == "v1.1"
    assert evidence["promotion"]["repository_promotion_ready"] is False
    assert evidence["promotion"]["public_release_ready"] is False
    assert evidence["promotion"]["decision"] == "blocked"
    assert all(
        gate["status"] in {"open", "blocked"} for gate in evidence["repository_gates"]
    )
    assert any(gate["status"] == "blocked" for gate in evidence["repository_gates"])
    assert all(gate["status"] == "open" for gate in evidence["human_gates"])


def test_external_distribution_gates_cover_every_binding_without_overclaim() -> None:
    evidence = _load(EVIDENCE_PATH)
    binding_matrix = _load(BINDING_PATH)
    by_id = {gate["binding_id"]: gate for gate in evidence["external_gates"]}

    assert set(by_id) == {binding["id"] for binding in binding_matrix["bindings"]}
    for binding in binding_matrix["bindings"]:
        gate = by_id[binding["id"]]
        assert gate["gate"] == binding["external_gate"]
        assert gate["status"] in {"not-started", "blocked"}
        assert gate["claim_policy"] == "no-distribution-claim-before-evidence"

    assert by_id["mojo"]["repository_prerequisite_complete"] is False
    assert by_id["rust"]["repository_prerequisite_complete"] is False
    assert by_id["python"]["repository_prerequisite_complete"] is True


def test_scientific_freeze_approval_is_preserved_but_not_overextended() -> None:
    evidence = _load(EVIDENCE_PATH)
    approval = evidence["scientific_freeze_approval"]

    assert approval["candidate_digest"] == (
        "9f437ea0b0521297b81f66adfac980e537db3c0ebf63823445f3bff2d285c3f9"
    )
    assert approval["reviewer"] == "edithatogo"
    assert approval["scope"] == "scientific-contract-only"
    assert approval["waives_implementation_or_release_gates"] is False
