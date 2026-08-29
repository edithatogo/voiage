"""Contracts for the dated dependency and preview-lane audit."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).parents[1]
AUDIT_PATH = (
    ROOT / "specs" / "submission-readiness" / "dependency-frontier-audit-20260829.json"
)


def _audit() -> dict[str, object]:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_dependency_audit_binds_the_refreshed_lock_and_declared_frontier() -> None:
    audit = _audit()
    assert audit["lock_refresh"]["resolved_packages"] == sum(
        line == "[[package]]" for line in (ROOT / "uv.lock").read_text().splitlines()
    )

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = len(config["project"]["dependencies"])
    declared += sum(
        len(items) for items in config["project"]["optional-dependencies"].values()
    )
    declared += len(config["build-system"]["requires"])
    assert audit["strict_frontier"]["declared_requirements"] == declared
    assert audit["strict_frontier"]["exit_status"] == 0
    assert audit["strict_frontier"]["policy_violations"] == 0
    assert audit["strict_frontier"]["locked_at_latest_compatible"] == declared


def test_preview_lanes_are_non_blocking_and_fail_closed_for_promotion() -> None:
    audit = _audit()
    lanes = audit["candidate_lanes"]
    assert len(lanes) == 5
    assert all(lane["candidates"] for lane in lanes)
    assert all(lane["promotion_gate"] for lane in lanes)
    assert {
        "non-blocking-preview",
        "measured-preview",
        "existing-experimental-extra",
    } <= {lane["mode"] for lane in lanes}

    findings = audit["findings"]
    assert {finding["id"] for finding in findings} == {
        f"DEP-{number:03d}" for number in range(1, 9)
    }
    states = {finding["id"]: finding["state"] for finding in findings}
    assert states["DEP-001"] == "resolved"
    assert states["DEP-003"] == "resolved"
    assert states["DEP-004"] == "resolved"
    assert states["DEP-005"] == "accepted_upstream_limitation"
    assert states["DEP-006"] == "resolved"
    assert states["DEP-007"] == "resolved"
    assert states["DEP-008"] == "resolved"
    assert states["DEP-002"] == "open"
    assert all(finding["required_disposition"] for finding in findings)
