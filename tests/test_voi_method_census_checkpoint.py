"""Contract tests for the residual VOI method census checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

CHECKPOINT = (
    Path(__file__).parents[1]
    / "conductor/archive/voi_method_census_contract_reconciliation_20260723"
    / "classification-checkpoint.json"
)
TRACK = CHECKPOINT.parent


def _checkpoint() -> dict[str, object]:
    return json.loads(CHECKPOINT.read_text(encoding="utf-8"))


def test_checkpoint_covers_exact_residual_census_scope() -> None:
    payload = _checkpoint()
    rows = payload["classifications"]
    assert isinstance(rows, list)
    assert [row["issue"] for row in rows] == [
        593,
        594,
        595,
        596,
        597,
        598,
        599,
        600,
        619,
    ]
    assert len({row["canonical_id"] for row in rows}) == 9


def test_checkpoint_requires_complete_classification_evidence() -> None:
    rows = _checkpoint()["classifications"]
    assert isinstance(rows, list)
    required = {
        "category",
        "classification_status",
        "functional",
        "assumptions",
        "compatibility",
        "maturity",
        "evidence_strength",
        "search_limit",
        "no_positive_claim",
        "sources",
    }
    for row in rows:
        assert required <= row.keys()
        assert len(row["assumptions"]) >= 5
        assert len(row["sources"]) >= 3
        assert row["no_positive_claim"]


def test_only_merged_dedicated_contracts_are_frozen_experimental() -> None:
    rows = _checkpoint()["classifications"]
    assert isinstance(rows, list)
    frozen = {
        row["issue"]
        for row in rows
        if row["classification_status"] == "frozen-experimental"
    }
    assert frozen == {595, 619}
    for row in rows:
        if row["issue"] not in frozen:
            assert row["classification_status"] == "candidate"
            assert row["maturity"] == "planned"


def test_checkpoint_never_promotes_candidate_classifications() -> None:
    payload = _checkpoint()
    boundary = payload["approval_boundary"]
    assert "named scientific and contract approval" in boundary
    assert "not stable or cross-language capability claims" in boundary


def test_phase_one_freeze_preserves_delivery_and_promotion_boundaries() -> None:
    """Classification completion must not promote the unfinished census."""
    plan = (TRACK / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    assert "- [x] **G3:** Freeze workstream estimands" in plan
    assert "- [x] **G4:** Run automated contract review" in plan
    assert "- **Migrated:** **G5:**" in plan
    assert "- **Migrated:** **G15:**" in plan
    assert "open #566 remains a" in plan
    assert gates["scientific-and-contract-review"]["status"] == "pending"
    assert gates["hosted-required-checks"]["status"] == "satisfied"
