"""Contracts for stable and preview dependency promotion policy."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
POLICY_PATH = (
    ROOT
    / "specs"
    / "submission-readiness"
    / "dependency-promotion-policy-20260829.json"
)
AUDIT_PATH = (
    ROOT / "specs" / "submission-readiness" / "dependency-frontier-audit-20260829.json"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_policy_covers_every_audited_candidate_without_changing_lane_identity() -> None:
    policy = _json(POLICY_PATH)
    audit = _json(AUDIT_PATH)
    lanes = {lane["id"]: lane for lane in policy["lanes"]}
    audited = {lane["id"]: lane for lane in audit["candidate_lanes"]}

    assert lanes.keys() == audited.keys()
    for lane_id, source in audited.items():
        assert lanes[lane_id]["source_lane"] == lane_id
        assert lanes[lane_id]["candidates"] == source["candidates"]
        assert lanes[lane_id]["required_gates"]
        assert lanes[lane_id]["isolation"]


def test_lifecycle_keeps_preview_observation_non_blocking() -> None:
    policy = _json(POLICY_PATH)
    lifecycle = {entry["stage"]: entry for entry in policy["lifecycle"]}

    assert list(lifecycle) == ["observe", "qualify", "candidate", "promote"]
    assert lifecycle["observe"]["blocking"] is False
    assert lifecycle["qualify"]["blocking"] is False
    assert lifecycle["candidate"]["blocking"] is True
    assert lifecycle["promote"]["blocking"] is True
    assert all(lane["blocking_stage"] == "candidate" for lane in policy["lanes"])


def test_universal_gate_requires_numerics_interchange_cpu_and_supply_chain() -> None:
    policy = _json(POLICY_PATH)
    gates = policy["universal_promotion_gates"]

    assert gates.keys() == {
        "correctness",
        "numerics",
        "interchange",
        "packaging",
        "security_and_licence",
        "performance",
        "documentation",
        "hosted",
    }
    assert policy["cpu_fallback_policy"]["required"] is True
    assert (
        "CPU-only clean install" in policy["cpu_fallback_policy"]["promotion_evidence"]
    )
    assert all(gates.values())


def test_receipts_and_rollback_are_fail_closed() -> None:
    policy = _json(POLICY_PATH)
    receipt = policy["promotion_receipt"]
    rollback = policy["rollback_policy"]

    assert len(receipt["required_fields"]) >= 20
    assert {"promote", "reject", "rollback"} <= set(receipt["allowed_decisions"])
    assert "numerical mismatch outside tolerance" in rollback["triggers"]
    assert "publish no release from the failed candidate" in rollback["actions"]
    assert rollback["data_compatibility"]
    assert len(policy["prohibited_shortcuts"]) == 6


def test_dependency_findings_have_explicit_policy_closure() -> None:
    closure = _json(POLICY_PATH)["finding_closure"]

    assert closure.keys() == {"DEP-001", "DEP-002", "DEP-003", "DEP-005", "DEP-006"}
    assert all(closure.values())
