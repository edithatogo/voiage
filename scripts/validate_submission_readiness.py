"""Validate the repository-wide submission-readiness contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

READINESS = {"published", "ready", "conditional", "blocked", "consideration", "retired"}
REQUIREMENT_STATUS = {"satisfied", "pending", "external", "not_applicable"}
TARGET_KINDS = {
    "archive",
    "community_review",
    "identifier",
    "journal",
    "package_registry",
    "sustainability",
}
PYOPENSCI_STATUSES = {"satisfied", "human_deferred"}
ROPENSCI_STATUSES = {"satisfied", "repository_blocked", "human_deferred"}


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_contract(path: Path, root: Path) -> dict[str, Any]:
    """Validate target coverage, evidence paths, and authority boundaries."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("submission contract must use schema_version 1.0")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("submission contract targets must be a non-empty array")

    identifiers: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise TypeError("submission target entries must be objects")
        identifier = _non_empty(target.get("id"), "target id")
        if identifier in identifiers:
            raise ValueError(f"duplicate submission target: {identifier}")
        identifiers.add(identifier)

        kind = target.get("kind")
        if kind not in TARGET_KINDS:
            raise ValueError(f"{identifier} has invalid kind: {kind}")
        readiness = target.get("readiness")
        if readiness not in READINESS:
            raise ValueError(f"{identifier} has invalid readiness: {readiness}")
        _non_empty(target.get("criteria_url"), f"{identifier} criteria_url")
        _non_empty(target.get("scope"), f"{identifier} scope")
        _non_empty(target.get("next_decision"), f"{identifier} next_decision")

        authority = target.get("authority")
        if not isinstance(authority, dict):
            raise TypeError(f"{identifier} authority must be an object")
        if authority.get("prepare") != "repository":
            raise ValueError(f"{identifier} preparation authority must be repository")
        if authority.get("submit") not in {"human", "external-system"}:
            raise ValueError(f"{identifier} submission authority must remain external")
        if authority.get("accept") != "external":
            raise ValueError(f"{identifier} acceptance authority must be external")

        acceptance_evidence = target.get("acceptance_evidence")
        if not isinstance(acceptance_evidence, list) or not acceptance_evidence:
            raise ValueError(f"{identifier} acceptance_evidence must be non-empty")
        for item in acceptance_evidence:
            _non_empty(item, f"{identifier} acceptance evidence")

        requirements = target.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(f"{identifier} requirements must be a non-empty array")
        requirement_ids: set[str] = set()
        unresolved = False
        for requirement in requirements:
            if not isinstance(requirement, dict):
                raise TypeError(f"{identifier} requirement must be an object")
            requirement_id = _non_empty(
                requirement.get("id"), f"{identifier} requirement id"
            )
            if requirement_id in requirement_ids:
                raise ValueError(
                    f"{identifier} has duplicate requirement: {requirement_id}"
                )
            requirement_ids.add(requirement_id)
            status = requirement.get("status")
            if status not in REQUIREMENT_STATUS:
                raise ValueError(
                    f"{identifier}/{requirement_id} has invalid status: {status}"
                )
            unresolved |= status in {"pending", "external"}
            evidence = requirement.get("evidence")
            if not isinstance(evidence, list):
                raise TypeError(
                    f"{identifier}/{requirement_id} evidence must be an array"
                )
            for relative in evidence:
                relative_path = Path(_non_empty(relative, "evidence path"))
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"unsafe evidence path: {relative}")
                if not (root / relative_path).exists():
                    raise ValueError(f"evidence path does not exist: {relative}")

        if readiness in {"ready", "published"} and unresolved:
            raise ValueError(f"{identifier} cannot be ready with an unmet gate")

    required = payload.get("required_target_ids")
    if not isinstance(required, list) or not set(required) <= identifiers:
        raise ValueError("required_target_ids must resolve to declared targets")

    criteria_refresh = payload.get("criteria_refresh")
    if not isinstance(criteria_refresh, dict):
        raise TypeError("criteria_refresh must be an object")
    _non_empty(criteria_refresh.get("reviewed_at"), "criteria_refresh reviewed_at")
    refresh_evidence = Path(
        _non_empty(criteria_refresh.get("evidence"), "criteria_refresh evidence")
    )
    if refresh_evidence.is_absolute() or ".." in refresh_evidence.parts:
        raise ValueError("criteria_refresh evidence path is unsafe")
    if not (root / refresh_evidence).exists():
        raise ValueError("criteria_refresh evidence path does not exist")
    refreshed_targets = criteria_refresh.get("target_ids")
    if not isinstance(refreshed_targets, list) or not set(required) <= set(
        refreshed_targets
    ):
        raise ValueError("criteria_refresh must cover every required target")

    lanes = payload.get("execution_lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("execution_lanes must be a non-empty array")
    lane_ids: set[str] = set()
    planned_targets: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            raise TypeError("submission execution lanes must be objects")
        lane_id = _non_empty(lane.get("id"), "execution lane id")
        if lane_id in lane_ids:
            raise ValueError(f"duplicate execution lane: {lane_id}")
        lane_ids.add(lane_id)
        issue_url = _non_empty(lane.get("issue_url"), f"{lane_id} issue_url")
        if not issue_url.startswith("https://github.com/edithatogo/voiage/issues/"):
            raise ValueError(f"{lane_id} issue_url must be a voiage GitHub issue")
        _non_empty(lane.get("repository_outcome"), f"{lane_id} repository_outcome")
        lane_targets = lane.get("targets")
        if not isinstance(lane_targets, list) or not lane_targets:
            raise ValueError(f"{lane_id} targets must be a non-empty array")
        unknown_targets = set(lane_targets) - identifiers
        if unknown_targets:
            raise ValueError(f"{lane_id} references unknown targets: {unknown_targets}")
        planned_targets.update(lane_targets)
    if not set(required) <= planned_targets:
        raise ValueError("every required target must belong to an execution lane")
    return {"target_count": len(targets), "targets": sorted(identifiers)}


def validate_pyopensci_evidence(path: Path, root: Path) -> dict[str, Any]:
    """Validate the repository-controlled pyOpenSci evidence matrix."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("pyOpenSci evidence must use schema_version 1.0")
    _non_empty(payload.get("criteria_url"), "pyOpenSci criteria_url")
    _non_empty(payload.get("reviewed_at"), "pyOpenSci reviewed_at")
    criteria = payload.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("pyOpenSci evidence criteria must be a non-empty array")
    criterion_ids: set[str] = set()
    human_deferred: set[str] = set()
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise TypeError("pyOpenSci criteria must be objects")
        identifier = _non_empty(criterion.get("id"), "pyOpenSci criterion id")
        if identifier in criterion_ids:
            raise ValueError(f"duplicate pyOpenSci criterion: {identifier}")
        criterion_ids.add(identifier)
        status = criterion.get("status")
        if status not in PYOPENSCI_STATUSES:
            raise ValueError(f"{identifier} has invalid pyOpenSci status: {status}")
        evidence = criterion.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{identifier} evidence must be a non-empty array")
        for relative in evidence:
            relative_path = Path(_non_empty(relative, "pyOpenSci evidence path"))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe pyOpenSci evidence path: {relative}")
            if not (root / relative_path).exists():
                raise ValueError(f"pyOpenSci evidence path does not exist: {relative}")
        if status == "human_deferred":
            human_deferred.add(identifier)
    required = payload.get("repository_controlled_criteria")
    if not isinstance(required, list) or not required:
        raise ValueError("repository_controlled_criteria must be a non-empty array")
    if any(
        not isinstance(identifier, str) or not identifier for identifier in required
    ):
        raise ValueError("repository_controlled_criteria must contain non-empty ids")
    if len(required) != len(set(required)) or not set(required) <= criterion_ids:
        raise ValueError("repository_controlled_criteria must resolve to criteria")
    unresolved_repository = {
        criterion["id"]
        for criterion in criteria
        if criterion["id"] in required and criterion["status"] != "satisfied"
    }
    if unresolved_repository:
        raise ValueError(
            "repository-controlled pyOpenSci criteria remain unresolved: "
            + ", ".join(sorted(unresolved_repository))
        )
    statuses = {criterion["id"]: criterion["status"] for criterion in criteria}
    if statuses.get("maintainer-commitment") != "satisfied":
        raise ValueError("pyOpenSci maintainer commitment must be recorded")
    if human_deferred != {"external-inquiry"}:
        raise ValueError("pyOpenSci human gates must remain explicit and bounded")
    return {"criterion_count": len(criteria), "deferred": sorted(human_deferred)}


def validate_ropensci_evidence(path: Path, root: Path) -> dict[str, Any]:
    """Validate that rOpenSci readiness evidence preserves unmet gates."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("rOpenSci evidence must use schema_version 1.0")
    criteria = payload.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("rOpenSci evidence criteria must be a non-empty array")
    statuses: dict[str, str] = {}
    for criterion in criteria:
        identifier = _non_empty(criterion.get("id"), "rOpenSci criterion id")
        if identifier in statuses:
            raise ValueError(f"duplicate rOpenSci criterion: {identifier}")
        status = criterion.get("status")
        if status not in ROPENSCI_STATUSES:
            raise ValueError(f"{identifier} has invalid rOpenSci status: {status}")
        statuses[identifier] = status
        evidence = criterion.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{identifier} evidence must be a non-empty array")
        for relative in evidence:
            relative_path = Path(_non_empty(relative, "rOpenSci evidence path"))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe rOpenSci evidence path: {relative}")
            if not (root / relative_path).exists():
                raise ValueError(f"rOpenSci evidence path does not exist: {relative}")
    if {key for key, value in statuses.items() if value == "repository_blocked"} != {
        "pkgcheck",
    }:
        raise ValueError("rOpenSci repository blockers must remain explicit")
    return {"criterion_count": len(criteria), "statuses": statuses}


def main() -> int:
    """Run validation from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "contract",
        nargs="?",
        type=Path,
        default=Path("specs/submission-readiness/targets.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    summary = validate_contract(arguments.contract, root)
    pyopensci = validate_pyopensci_evidence(
        root / "specs/submission-readiness/pyopensci-evidence.json", root
    )
    ropensci = validate_ropensci_evidence(
        root / "specs/submission-readiness/ropensci-evidence.json", root
    )
    print(
        "Submission readiness contract: PASS "
        f"({summary['target_count']} targets; {pyopensci['criterion_count']} "
        f"pyOpenSci criteria; {ropensci['criterion_count']} rOpenSci criteria)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
