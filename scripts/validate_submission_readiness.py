"""Validate the repository-wide submission-readiness contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

ROOT_REFRESH_PLACEHOLDER = "REPLACE_AFTER_FINAL_ROOT_PR_MERGES"
CURRENT_ISSUE_LANES = {
    "contract": 614,
    "paper_and_author_boundaries": 296,
    "python_community_review": 1037,
    "r_community_and_journal_readiness": 615,
    "distinct_publication_and_sustainability_assessment": 1026,
    "hpc_distribution": 1025,
}
HISTORICAL_COMPLETED_ISSUE_LANES = {
    "paper_and_author_boundaries": 299,
    "python_community_repository_readiness": 616,
    "distinct_publication_and_sustainability_assessment": 617,
    "initial_hpc_recipe_contract": 622,
}
EXPECTED_EXECUTION_LANE_ISSUES = {
    "contract-maintenance": 614,
    "paper-and-author-boundaries": 296,
    "python-community-review": 1037,
    "r-community-and-journal-readiness": 615,
    "distinct-publication-and-sustainability-assessment": 1026,
    "hpc-distribution-readiness": 1025,
}
EXPECTED_RELEASE_EVIDENCE = {
    "version": "2.2.0",
    "release_commit": "7af563c8cb373057d30662650b3f332f39e05b83",
    "github_and_pypi_published": True,
}
EXPECTED_FINAL_ROOT_BASE = "fea90d41898ac31c970b0c2b7a8a80ef3366ab96"
EXPECTED_FINAL_ROOT_PR = {
    "number": 1087,
    "head_sha": "614224cf4cab2514ece333345ff25f7441fddeba",
    "merge_sha": EXPECTED_FINAL_ROOT_BASE,
    "reviewed_tree": "44a79dedb8cf4c8ce6a62f12d549bb6e7585b2ef",
    "merged_tree": "44a79dedb8cf4c8ce6a62f12d549bb6e7585b2ef",
    "tree_equal": True,
    "terminal_checks": 40,
}
EXPECTED_EXTERNAL_OUTCOMES = {
    "pyopensci_acceptance": "pending_external",
    "joss_acceptance": "pending_external",
    "arxiv_announcement": "pending_external",
    "spack_upstream_acceptance": "pending_external",
    "easybuild_upstream_acceptance": "pending_external",
    "binarybuilder_jll_acceptance": "pending_external",
    "julia_general_indexing": "pending_external",
}

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
ROPENSCI_STATUSES = {
    "satisfied",
    "repository_blocked",
    "hosted_pending",
    "human_deferred",
}
R_DISTRIBUTION_RECEIPT = Path(
    "specs/submission-readiness/r-distribution-evidence-20260902.json"
)
R_DISTRIBUTION_TESTED_HEAD = "78c4514f1dfe91b5ce4892ccce1b6f742d500da0"
R_DISTRIBUTION_CURRENT_REVISION = "279cc50459453c78c8602d1e51a9a5a6f5025165"
R_DISTRIBUTION_JOBS = {
    "Cross-language differential conformance": 99584273115,
    "Julia binding (1.10, macos-latest, rust/target/release/libvoiage_ffi.dylib)": 99582630629,
    "Julia binding (1.10, ubuntu-latest, rust/target/release/libvoiage_ffi.so)": 99582630843,
    "Julia binding (1.10, windows-latest, rust/target/release/voiage_ffi.dll)": 99582630648,
    "Julia binding (1.11, macos-latest, rust/target/release/libvoiage_ffi.dylib)": 99582630694,
    "Julia binding (1.11, ubuntu-latest, rust/target/release/libvoiage_ffi.so)": 99582630953,
    "Julia binding (1.11, windows-latest, rust/target/release/voiage_ffi.dll)": 99582630743,
    "Julia binding (1.12, macos-latest, rust/target/release/libvoiage_ffi.dylib)": 99582630914,
    "Julia binding (1.12, ubuntu-latest, rust/target/release/libvoiage_ffi.so)": 99582630824,
    "Julia binding (1.12, windows-latest, rust/target/release/voiage_ffi.dll)": 99582630788,
    "R installed native smoke (macos-latest)": 99582630716,
    "R installed native smoke (ubuntu-latest)": 99582630515,
    "R installed native smoke (windows-latest)": 99582630718,
    "R package-development checks": 99582630624,
    "Rust workspace": 99582630399,
    "Rust workspace (MSRV 1.85)": 99582630200,
}
R_DISTRIBUTION_OBJECTS = {
    ".github/workflows/bindings-ci.yml": "35ed854f290b5c841f200f720661d44792922ba9",
    "r-package/voiageR": "4bcce19b910a1a4404668538ce845fe39a1079c5",
    "rust/crates/voiage-ffi": "b9f6728a6f7ae5ee8ef6018ca4dbbce404861266",
    "specs/numerical-reference": "bb3ca823420fbe8f95b85b9c92ce07b3988cea2b",
}
R_MANUAL_RECEIPT = {
    "path": "conductor/tracks/remaining_backlog_delivery_20260831/r-manual-check-20260901.json",
    "sha256": "520af27e4d3463d9e9404c06e64e86d5a38a9db66bb3cb77e8338ac014fdf3a7",
}
R_ARCHIVE_SHA256 = "af485e1cfba6dc9c1f149ce074640c8ef63bb3f42c649f71fccbe0e5d114c8e4"


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object(root: Path, revision: str, path: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ValueError("git is required to validate R distribution inputs")
    result = subprocess.run(  # noqa: S603 - revisions and paths are pinned constants
        [git, "rev-parse", "--verify", f"{revision}:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"cannot resolve recorded R distribution input: {revision}:{path}"
        )
    return result.stdout.strip()


def validate_r_distribution_evidence(path: Path, root: Path) -> dict[str, Any]:
    """Validate current R distribution evidence without inventing venue outcomes."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "voiage.r-distribution-evidence.v1"
        or payload.get("state")
        != "current_repository_evidence_external_actions_unperformed"
        or payload.get("current_revision") != R_DISTRIBUTION_CURRENT_REVISION
    ):
        raise ValueError("R distribution evidence identity or state is invalid")

    hosted = payload.get("hosted_run")
    if not isinstance(hosted, dict) or {
        "workflow": hosted.get("workflow"),
        "run_id": hosted.get("run_id"),
        "url": hosted.get("url"),
        "head_sha": hosted.get("head_sha"),
        "conclusion": hosted.get("conclusion"),
    } != {
        "workflow": "R CMD Check and Retained Bindings CI",
        "run_id": 33420870772,
        "url": "https://github.com/edithatogo/voiage/actions/runs/33420870772",
        "head_sha": R_DISTRIBUTION_TESTED_HEAD,
        "conclusion": "success",
    }:
        raise ValueError("R distribution hosted run binding is invalid")
    jobs = hosted.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != len(R_DISTRIBUTION_JOBS)
        or any(not isinstance(job, dict) for job in jobs)
        or {job.get("name"): job.get("database_id") for job in jobs}
        != R_DISTRIBUTION_JOBS
        or any(job.get("conclusion") != "success" for job in jobs)
    ):
        raise ValueError("R distribution hosted job binding is invalid")

    equality = payload.get("tested_input_equality")
    entries = equality.get("paths") if isinstance(equality, dict) else None
    if (
        not isinstance(equality, dict)
        or equality.get("method") != "git_object_identity"
        or not isinstance(entries, list)
        or any(not isinstance(entry, dict) for entry in entries)
        or {
            entry.get("path"): (
                entry.get("tested_head_object_id"),
                entry.get("current_revision_object_id"),
            )
            for entry in entries
        }
        != {
            path: (object_id, object_id)
            for path, object_id in R_DISTRIBUTION_OBJECTS.items()
        }
    ):
        raise ValueError("R distribution tested-input inventory is invalid")
    for relative, expected in R_DISTRIBUTION_OBJECTS.items():
        # Hosted pull-request checkouts are deliberately shallow and need not
        # contain either recorded commit.  The receipt and constants pin both
        # observations; resolve the exact checked-out candidate independently
        # so a later path mutation cannot inherit the equality claim.
        if _git_object(root, "HEAD", relative) != expected:
            raise ValueError("R distribution tested inputs do not match")

    archive = payload.get("source_archive")
    binding = archive.get("manual_check_receipt") if isinstance(archive, dict) else None
    if (
        not isinstance(archive, dict)
        or archive.get("filename") != "voiageR_2.2.0.tar.gz"
        or archive.get("sha256") != R_ARCHIVE_SHA256
        or binding != R_MANUAL_RECEIPT
    ):
        raise ValueError("R source archive or manual receipt binding is invalid")
    manual_path = root / R_MANUAL_RECEIPT["path"]
    if not manual_path.is_file() or _sha256(manual_path) != R_MANUAL_RECEIPT["sha256"]:
        raise ValueError("R manual-check receipt bytes do not match")
    manual: Any = json.loads(manual_path.read_text(encoding="utf-8"))
    expected_notes = [
        "CRAN incoming feasibility: New submission",
        "future file timestamps: unable to verify current time",
    ]
    if (
        not isinstance(manual, dict)
        or manual.get("package_errors") != 0
        or manual.get("package_warnings") != 0
        or manual.get("notes") != expected_notes
        or manual.get("strict_zero_note_criterion_met") is not False
        or manual.get("check_suppressions") != []
        or manual.get("cran_submission") is not False
        or manual.get("artifacts", {}).get("voiageR_2.2.0.tar.gz", {}).get("sha256")
        != R_ARCHIVE_SHA256
    ):
        raise ValueError(
            "R manual-check receipt does not preserve the reviewed outcome"
        )
    if payload.get("check_outcome") != {
        "errors": 0,
        "warnings": 0,
        "notes": expected_notes,
        "strict_zero_note_criterion_met": False,
        "check_suppressions": [],
        "interpretation": "Both NOTEs are retained as external or environment observations; they are not suppressed or relabelled as package defects.",
    }:
        raise ValueError("R check outcome is inconsistent with the manual receipt")
    if payload.get("review_packet") != {
        "path": "docs/release/ropensci-presubmission-inquiry-draft.md",
        "state": "prepared_local_unposted",
    } or payload.get("external_actions") != {
        "cran_submission": False,
        "ropensci_inquiry": False,
        "ropensci_submission": False,
    }:
        raise ValueError("R review packet must remain local and external actions false")
    if payload.get("external_outcomes") != {
        "cran_acceptance": "pending_external",
        "ropensci_scope_review_acceptance": "pending_external",
    }:
        raise ValueError("R distribution external outcomes must remain pending")
    return {"job_count": len(jobs), "input_count": len(entries)}


def validate_contract(path: Path, root: Path) -> dict[str, Any]:
    """Validate target coverage, evidence paths, and authority boundaries."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("submission contract must use schema_version 1.0")

    evidence_refresh = payload.get("evidence_refresh")
    if not isinstance(evidence_refresh, dict):
        raise TypeError("evidence_refresh must be an object")
    _non_empty(evidence_refresh.get("reviewed_at"), "evidence_refresh reviewed_at")
    refresh_path = Path(
        _non_empty(evidence_refresh.get("evidence"), "evidence_refresh evidence")
    )
    if refresh_path.is_absolute() or ".." in refresh_path.parts:
        raise ValueError("evidence_refresh evidence path is unsafe")
    if not (root / refresh_path).is_file():
        raise ValueError("evidence_refresh evidence path does not exist")
    refresh_record: Any = json.loads((root / refresh_path).read_text(encoding="utf-8"))
    expected_refresh_sha = evidence_refresh.get("evidence_sha256")
    actual_refresh_sha = hashlib.sha256((root / refresh_path).read_bytes()).hexdigest()
    if expected_refresh_sha != actual_refresh_sha:
        raise ValueError("evidence_refresh record SHA-256 does not match")
    if (
        not isinstance(refresh_record, dict)
        or refresh_record.get("schema_version")
        != "voiage.cross-venue-evidence-refresh.v1"
        or refresh_record.get("reviewed_at") != evidence_refresh.get("reviewed_at")
        or refresh_record.get("state") != evidence_refresh.get("state")
        or refresh_record.get("final_root_pr") != evidence_refresh.get("final_root_pr")
    ):
        raise ValueError("evidence_refresh record binding is invalid")
    if refresh_record.get("current_issue_lanes") != CURRENT_ISSUE_LANES:
        raise ValueError("evidence_refresh current issue lanes are stale")
    if (
        refresh_record.get("historical_completed_issue_lanes")
        != HISTORICAL_COMPLETED_ISSUE_LANES
    ):
        raise ValueError("evidence_refresh historical issue lanes are stale")
    if refresh_record.get("release") != EXPECTED_RELEASE_EVIDENCE:
        raise ValueError("evidence_refresh release evidence is stale")
    if refresh_record.get("external_outcomes") != EXPECTED_EXTERNAL_OUTCOMES:
        raise ValueError("evidence_refresh external outcomes must remain pending")
    repository_state = refresh_record.get("repository_state")
    expected_repository_state = {
        "pyopensci_declarations_confirmed": True,
        "pyopensci_survey_completed": False,
        "pyopensci_human_written_submission_supplied": False,
        "pyopensci_submission_performed": False,
        "joss_submission_performed": False,
        "arxiv_submission_currently_verified": False,
        "spack_recipe_prepared": True,
        "spack_current_native_build_complete": False,
        "spack_upstream_submission_performed": False,
        "easybuild_provider_recipes_prepared": True,
        "easybuild_final_root_graph_merged": evidence_refresh.get("state")
        == "complete",
        "easybuild_native_foss_builds_complete": False,
        "easybuild_upstream_submission_performed": False,
        "yggdrasil_v2_2_update_submitted": False,
        "binarybuilder_jll_accepted": False,
        "julia_general_registration_submitted": False,
        "julia_general_indexed": False,
    }
    if repository_state != expected_repository_state:
        raise ValueError("evidence_refresh repository state is incomplete or stale")
    base_main = refresh_record.get("base_main")
    if (
        not isinstance(base_main, str)
        or len(base_main) != 40
        or any(character not in "0123456789abcdef" for character in base_main)
    ):
        raise ValueError("evidence_refresh base_main must be an exact commit")
    root_pr = evidence_refresh.get("final_root_pr")
    if not isinstance(root_pr, dict):
        raise TypeError("evidence_refresh final_root_pr must be an object")
    if evidence_refresh.get("state") == "awaiting_final_root_pr_merge":
        if root_pr != {
            "number": None,
            "head_sha": None,
            "merge_sha": None,
            "reviewed_tree": None,
            "merged_tree": None,
            "tree_equal": None,
            "terminal_checks": None,
            "placeholder": ROOT_REFRESH_PLACEHOLDER,
        }:
            raise ValueError("pending final-root refresh must retain exact placeholder")
    elif evidence_refresh.get("state") == "complete":
        hex_fields = ("head_sha", "merge_sha", "reviewed_tree", "merged_tree")
        if (
            not isinstance(root_pr.get("number"), int)
            or any(
                not isinstance(root_pr.get(field), str)
                or len(root_pr[field]) != 40
                or any(
                    character not in "0123456789abcdef" for character in root_pr[field]
                )
                for field in hex_fields
            )
            or root_pr.get("reviewed_tree") != root_pr.get("merged_tree")
            or root_pr.get("tree_equal") is not True
            or not isinstance(root_pr.get("terminal_checks"), int)
            or root_pr["terminal_checks"] < 1
            or "placeholder" in root_pr
        ):
            raise ValueError(
                "complete final-root refresh lacks authoritative merge evidence"
            )
        if root_pr != EXPECTED_FINAL_ROOT_PR or base_main != EXPECTED_FINAL_ROOT_BASE:
            raise ValueError("complete final-root refresh is not bound to PR #1087")
    else:
        raise ValueError("evidence_refresh state must be pending or complete")
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
    refreshed_evidence_targets = evidence_refresh.get("target_ids")
    if not isinstance(refreshed_evidence_targets, list) or not set(required) <= set(
        refreshed_evidence_targets
    ):
        raise ValueError("evidence_refresh must cover every required target")

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
        expected_issue = EXPECTED_EXECUTION_LANE_ISSUES.get(lane_id)
        if expected_issue is None or issue_url != (
            f"https://github.com/edithatogo/voiage/issues/{expected_issue}"
        ):
            raise ValueError(f"{lane_id} must route to its current open issue")
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
    if lane_ids != set(EXPECTED_EXECUTION_LANE_ISSUES):
        raise ValueError("execution lanes must match the current issue routes")
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
    blocked = {
        key
        for key, value in statuses.items()
        if value in {"repository_blocked", "hosted_pending"}
    }
    if blocked not in (set(), {"pkgcheck"}):
        raise ValueError("rOpenSci repository or hosted blockers must remain explicit")
    distribution = validate_r_distribution_evidence(root / R_DISTRIBUTION_RECEIPT, root)
    return {
        "criterion_count": len(criteria),
        "statuses": statuses,
        "distribution": distribution,
    }


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
