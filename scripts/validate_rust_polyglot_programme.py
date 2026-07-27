#!/usr/bin/env python3
"""Validate the Rust-first polyglot programme governance topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


PROJECT_URL = "https://github.com/users/edithatogo/projects/28"
PARENT_TRACK = "rust_polyglot_voi_completion_20260723"
PARENT_ISSUE = 313
TRACK_ISSUES: dict[str, int] = {
    PARENT_TRACK: PARENT_ISSUE,
    "voi_method_census_contract_reconciliation_20260723": 314,
    "external_voi_library_feature_parity_20260723": 315,
    "stable_voi_rust_core_completion_20260723": 316,
    "value_of_perspective_completion_20260723": 317,
    "supported_frontier_method_completion_20260723": 318,
    "ml_llm_agent_voi_20260723": 319,
    "polyglot_abi_binding_parity_20260723": 320,
    "datasets_worked_examples_20260723": 321,
    "quality_release_automation_20260723": 322,
    "research_contribution_ai_transparency_20260723": 323,
}
EXPECTED_PROJECT: dict[int, dict[str, str]] = {
    313: {
        "priority": "P0",
        "risk level": "High",
        "review due": "2026-08-31",
        "status": "In Progress",
    },
    314: {
        "priority": "P0",
        "risk level": "Medium",
        "review due": "2026-08-15",
        "status": "In Progress",
    },
    315: {
        "priority": "P0",
        "risk level": "High",
        "review due": "2026-08-31",
        "status": "In Progress",
    },
    316: {
        "priority": "P0",
        "risk level": "High",
        "review due": "2026-08-31",
        "status": "In Progress",
    },
    317: {"priority": "P1", "risk level": "High", "review due": "2026-09-30"},
    318: {"priority": "P1", "risk level": "Medium", "review due": "2026-09-30"},
    319: {"priority": "P1", "risk level": "High", "review due": "2026-10-31"},
    320: {"priority": "P1", "risk level": "High", "review due": "2026-10-31"},
    321: {"priority": "P2", "risk level": "Medium", "review due": "2026-10-31"},
    322: {
        "priority": "P1",
        "risk level": "High",
        "review due": "2026-10-31",
        "status": "In Progress",
    },
    323: {"priority": "P1", "risk level": "High", "review due": "2026-08-31"},
}
EXPECTED_PROJECT_VIEWS: dict[str, dict[str, str]] = {
    "Current Delivery": {
        "layout": "TABLE_LAYOUT",
        "filter": 'status:"In Progress"',
    },
    "Next: Software Landscape": {
        "layout": "TABLE_LAYOUT",
        "filter": 'track-id:"external_voi_library_feature_parity_20260723"',
    },
    "MoSCoW & Priority": {
        "layout": "BOARD_LAYOUT",
        "filter": "",
    },
    "Industry & Adoption": {
        "layout": "TABLE_LAYOUT",
        "filter": 'record-type:"Development ledger"',
    },
    "Gates & High Risk": {
        "layout": "TABLE_LAYOUT",
        "filter": "risk-level:High",
    },
    "Evidence & Review Due": {
        "layout": "TABLE_LAYOUT",
        "filter": "evidence-state:Unverified",
    },
}
FRONTIER_TRACK = "supported_frontier_method_completion_20260723"
FRONTIER_PARENT_ISSUE = TRACK_ISSUES[FRONTIER_TRACK]
FRONTIER_METHOD_GAP_SUBISSUES: dict[int, dict[str, str]] = {
    556: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "deterministic-sensitivity-analysis",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "Medium",
        "gate": "Local",
        "contract version": "1.0.0",
        "review due": "2026-09-30",
    },
    557: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "value-of-distributional-information",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.0.0",
        "review due": "2026-09-30",
    },
    558: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "qualitative-voi",
        "moscow": "Should",
        "priority": "P2",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.0.0",
        "review due": "2026-10-31",
    },
    559: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "value-of-flexibility",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "Medium",
        "gate": "Local",
        "contract version": "1.0.0",
        "review due": "2026-09-30",
    },
    560: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "mcda-voi",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.0.0",
        "review due": "2026-09-30",
    },
}
INDUSTRY_SUBISSUES: dict[int, dict[str, str]] = {
    565: {
        "track id": "external_voi_library_feature_parity_20260723",
        "parent issue": "315",
        "record id": "landscape-open-source-inventory",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "Medium",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-08-15",
    },
    566: {
        "track id": "voi_method_census_contract_reconciliation_20260723",
        "parent issue": "314",
        "record id": "industry-decision-problem-contract",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-09-30",
    },
    567: {
        "track id": "external_voi_library_feature_parity_20260723",
        "parent issue": "315",
        "record id": "landscape-gap-review-roadmap-proposal",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Human",
        "contract version": "1.1.0",
        "review due": "2026-08-31",
    },
    568: {
        "track id": "external_voi_library_feature_parity_20260723",
        "parent issue": "315",
        "record id": "landscape-commercial-hosted-inventory",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "External",
        "contract version": "1.1.0",
        "review due": "2026-08-15",
    },
    569: {
        "track id": "external_voi_library_feature_parity_20260723",
        "parent issue": "315",
        "record id": "landscape-schema-review-protocol",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-08-15",
    },
    570: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "risk-sensitive-constrained-voi",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-09-30",
    },
    571: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "experiment-portfolio-voi",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    572: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "forecast-signal-information-voi",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "Medium",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    573: {
        "track id": "external_voi_library_feature_parity_20260723",
        "parent issue": "315",
        "record id": "landscape-capability-adoption-map",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "Medium",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-08-31",
    },
    574: {
        "track id": "datasets_worked_examples_20260723",
        "parent issue": "321",
        "record id": "churn-retention-policy-voi-example",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "External",
        "contract version": "1.1.0",
        "review due": "2026-09-30",
    },
    575: {
        "track id": "datasets_worked_examples_20260723",
        "parent issue": "321",
        "record id": "industry-domain-example-packs",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "Medium",
        "gate": "External",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    576: {
        "track id": "ml_llm_agent_voi_20260723",
        "parent issue": "319",
        "record id": "decision-focused-model-value",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    577: {
        "track id": "datasets_worked_examples_20260723",
        "parent issue": "321",
        "record id": "domain-template-adapter-registry",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "Medium",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    578: {
        "track id": "ml_llm_agent_voi_20260723",
        "parent issue": "319",
        "record id": "policy-uplift-voi",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-09-30",
    },
    579: {
        "track id": "polyglot_abi_binding_parity_20260723",
        "parent issue": "320",
        "record id": "industry-decision-contract-binding-parity",
        "moscow": "Must",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    580: {
        "track id": "quality_release_automation_20260723",
        "parent issue": "322",
        "record id": "decision-registry-cards",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Human",
        "contract version": "1.1.0",
        "review due": "2026-09-30",
    },
    581: {
        "track id": "quality_release_automation_20260723",
        "parent issue": "322",
        "record id": "local-decision-studio-reporting",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    582: {
        "track id": FRONTIER_TRACK,
        "parent issue": str(FRONTIER_PARENT_ISSUE),
        "record id": "information-source-portfolio-voi",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    583: {
        "track id": "quality_release_automation_20260723",
        "parent issue": "322",
        "record id": "enterprise-integration-adapters",
        "moscow": "Should",
        "priority": "P1",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-10-31",
    },
    584: {
        "track id": "quality_release_automation_20260723",
        "parent issue": "322",
        "record id": "decision-correctness-industry-assurance",
        "moscow": "Must",
        "priority": "P0",
        "risk level": "High",
        "gate": "Local",
        "contract version": "1.1.0",
        "review due": "2026-09-30",
    },
}
FRONTIER_SUBISSUES = {
    **FRONTIER_METHOD_GAP_SUBISSUES,
    **{
        issue_number: fields
        for issue_number, fields in INDUSTRY_SUBISSUES.items()
        if fields["track id"] == FRONTIER_TRACK
    },
}
SUBISSUE_GROUPS: dict[str, dict[str, Any]] = {
    track_id: {
        "parent issue": TRACK_ISSUES[track_id],
        "issues": {
            issue_number: fields
            for issue_number, fields in {
                **FRONTIER_METHOD_GAP_SUBISSUES,
                **INDUSTRY_SUBISSUES,
            }.items()
            if fields["track id"] == track_id
        },
    }
    for track_id in (
        "voi_method_census_contract_reconciliation_20260723",
        "external_voi_library_feature_parity_20260723",
        FRONTIER_TRACK,
        "ml_llm_agent_voi_20260723",
        "polyglot_abi_binding_parity_20260723",
        "datasets_worked_examples_20260723",
        "quality_release_automation_20260723",
    )
}
REQUIRED_FILES = ("spec.md", "plan.md", "metadata.json", "index.md", "evidence.jsonl")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _normalize_fields(item: Mapping[str, object]) -> dict[str, object]:
    return {str(key).casefold(): value for key, value in item.items()}


def _local_dependency_graph(repo: Path) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for track_id in TRACK_ISSUES:
        metadata = _load_json(
            repo / "conductor" / "tracks" / track_id / "metadata.json"
        )
        graph[track_id] = {
            dependency
            for dependency in metadata.get("dependencies", [])
            if dependency in TRACK_ISSUES
        }
    return graph


def _validate_acyclic(graph: Mapping[str, set[str]]) -> list[str]:
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: tuple[str, ...]) -> None:
        if node in visiting:
            errors.append("dependency cycle: " + " -> ".join((*trail, node)))
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, set()):
            visit(dependency, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for track_id in graph:
        visit(track_id, ())
    return errors


def validate_local(repo: Path) -> list[str]:
    """Return local governance validation errors."""
    errors: list[str] = []
    registry = (repo / "conductor" / "tracks.md").read_text(encoding="utf-8")

    for track_id, issue in TRACK_ISSUES.items():
        root = repo / "conductor" / "tracks" / track_id
        for filename in REQUIRED_FILES:
            path = root / filename
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                errors.append(f"{track_id}: missing or empty {filename}")
        if not root.is_dir():
            continue

        metadata = _load_json(root / "metadata.json")
        expected_url = f"https://github.com/edithatogo/voiage/issues/{issue}"
        required_metadata = (
            "track_id",
            "version",
            "type",
            "status",
            "created_at",
            "updated_at",
            "description",
            "evidence_schema",
            "github_issue",
            "github_project",
            "dependencies",
            "gates",
        )
        errors.extend(
            f"{track_id}: metadata missing {key}"
            for key in required_metadata
            if key not in metadata
        )
        if metadata.get("track_id") != track_id:
            errors.append(f"{track_id}: metadata track_id mismatch")
        if metadata.get("github_issue") != expected_url:
            errors.append(f"{track_id}: GitHub issue mismatch")
        if metadata.get("github_project") != PROJECT_URL:
            errors.append(f"{track_id}: GitHub project mismatch")
        if metadata.get("evidence_schema") != "1.0":
            errors.append(f"{track_id}: evidence schema must be 1.0")
        if f"./tracks/{track_id}/index.md" not in registry:
            errors.append(f"{track_id}: registry link missing")

        spec = (root / "spec.md").read_text(encoding="utf-8")
        plan = (root / "plan.md").read_text(encoding="utf-8")
        index = (root / "index.md").read_text(encoding="utf-8")
        if "# Track Specification:" not in spec or "Acceptance criteria" not in spec:
            errors.append(f"{track_id}: specification contract incomplete")
        required_plan_tokens = (
            "# Track Implementation Plan:",
            "Phase 1:",
            "Phase 2:",
            "Phase 3:",
            "git note",
            "short commit SHA",
            "plan update",
            "Conductor - User Manual Verification",
        )
        errors.extend(
            f"{track_id}: plan missing {token}"
            for token in required_plan_tokens
            if token not in plan
        )
        index_files = ("spec.md", "plan.md", "metadata.json", "evidence.jsonl")
        errors.extend(
            f"{track_id}: index missing {filename}"
            for filename in index_files
            if f"(./{filename})" not in index
        )
        if track_id in SUBISSUE_GROUPS:
            subissues = SUBISSUE_GROUPS[track_id]["issues"]
            expected_subissues = {
                f"https://github.com/edithatogo/voiage/issues/{issue_number}"
                for issue_number in subissues
            }
            observed_subissues = set(metadata.get("github_subissues", []))
            if observed_subissues != expected_subissues:
                errors.append(
                    f"{track_id}: GitHub subissues mismatch; expected "
                    f"{sorted(expected_subissues)}, got {sorted(observed_subissues)}"
                )
            for issue_number, fields in subissues.items():
                tokens = (
                    f"#{issue_number}",
                    fields["record id"],
                )
                for token in tokens:
                    if token not in spec:
                        errors.append(f"{track_id}: specification missing {token}")
                    if token not in plan:
                        errors.append(f"{track_id}: plan missing {token}")
                if f"issues/{issue_number}" not in index:
                    errors.append(f"{track_id}: index missing issue #{issue_number}")

        for line_number, line in enumerate(
            (root / "evidence.jsonl").read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{track_id}: invalid evidence JSONL line {line_number}")
                continue
            if record.get("schema_version") != "1.0":
                errors.append(f"{track_id}: invalid evidence schema line {line_number}")

    errors.extend(_validate_acyclic(_local_dependency_graph(repo)))
    return errors


def _run_json(command: list[str], repo: Path) -> Any:
    completed = subprocess.run(  # noqa: S603 - command is repository-owned
        command,
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def missing_required_subissues(observed: set[int]) -> set[int]:
    """Return required programme subissues absent from a live parent."""
    expected = set(TRACK_ISSUES.values()) - {PARENT_ISSUE}
    return expected - observed


def missing_frontier_subissues(observed: set[int]) -> set[int]:
    """Return required frontier gap subissues absent from issue #318."""
    return set(FRONTIER_SUBISSUES) - observed


def missing_track_subissues(track_id: str, observed: set[int]) -> set[int]:
    """Return required native subissues absent from a governed track parent."""
    group = SUBISSUE_GROUPS[track_id]
    return set(group["issues"]) - observed


def validate_live_github(repo: Path) -> list[str]:
    """Return live GitHub issue, subissue, and Project 28 validation errors."""
    errors: list[str] = []
    subissue_queries = " ".join(
        f"group{index}: issue(number:{group['parent issue']}) {{ number "
        "subIssues(first:100) { nodes { number } } }"
        for index, group in enumerate(SUBISSUE_GROUPS.values())
    )
    query = (
        'query { repository(owner:"edithatogo",name:"voiage") { '
        "programme: issue(number:313) { number "
        "subIssues(first:100) { nodes { number } } } "
        f"{subissue_queries} }} }}"
    )
    result = _run_json(["gh", "api", "graphql", "-f", f"query={query}"], repo)
    repository = result["data"]["repository"]
    subissues = {
        node["number"]
        for node in repository["programme"]["subIssues"]["nodes"]
    }
    missing_subissues = missing_required_subissues(subissues)
    if missing_subissues:
        errors.append(
            "required native subissues missing: "
            f"{sorted(missing_subissues)}; got {sorted(subissues)}"
        )
    for index, (track_id, _group) in enumerate(SUBISSUE_GROUPS.items()):
        observed = {
            node["number"]
            for node in repository[f"group{index}"]["subIssues"]["nodes"]
        }
        missing = missing_track_subissues(track_id, observed)
        if missing:
            errors.append(
                f"{track_id}: required native subissues missing: "
                f"{sorted(missing)}; got {sorted(observed)}"
            )

    issue_data = _run_json(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            "edithatogo/voiage",
            "--state",
            "all",
            "--limit",
            "400",
            "--json",
            "number,body,state",
        ],
        repo,
    )
    issues = {item["number"]: item for item in issue_data}
    for track_id, issue_number in TRACK_ISSUES.items():
        issue = issues.get(issue_number)
        if issue is None:
            errors.append(f"{track_id}: GitHub issue missing")
            continue
        body = issue.get("body") or ""
        if issue.get("state") != "OPEN":
            errors.append(f"{track_id}: issue is not open")
        required_issue_tokens = (
            "<!-- voiage-conductor-managed:start -->",
            f"Track ID: {track_id}",
            "<!-- voiage-conductor-managed:end -->",
            "closure",
        )
        errors.extend(
            f"{track_id}: issue body missing {token}"
            for token in required_issue_tokens
            if token.casefold() not in body.casefold()
        )
    governed_subissues = {
        issue_number: fields
        for group in SUBISSUE_GROUPS.values()
        for issue_number, fields in group["issues"].items()
    }
    for issue_number, fields in governed_subissues.items():
        track_id = fields["track id"]
        issue = issues.get(issue_number)
        if issue is None:
            errors.append(f"{track_id}: GitHub subissue #{issue_number} missing")
            continue
        body = issue.get("body") or ""
        if issue.get("state") != "OPEN":
            errors.append(f"{track_id}: subissue #{issue_number} is not open")
        record_token = (
            f"Method family: `{fields['record id']}`"
            if issue_number in FRONTIER_METHOD_GAP_SUBISSUES
            else f"Record ID: `{fields['record id']}`"
        )
        required_issue_tokens = (
            "<!-- voiage-conductor-managed:start -->",
            track_id,
            f"Parent issue: #{fields['parent issue']}",
            record_token,
            "<!-- voiage-conductor-managed:end -->",
            "closure",
        )
        errors.extend(
            f"{track_id}: subissue #{issue_number} body missing {token}"
            for token in required_issue_tokens
            if token.casefold() not in body.casefold()
        )

    project = _run_json(
        [
            "gh",
            "project",
            "item-list",
            "28",
            "--owner",
            "edithatogo",
            "--limit",
            "500",
            "--format",
            "json",
        ],
        repo,
    )
    items = {
        item["content"]["number"]: _normalize_fields(item)
        for item in project["items"]
        if isinstance(item.get("content"), dict)
        and item["content"].get("repository") == "edithatogo/voiage"
        and item["content"].get("number")
        in {*TRACK_ISSUES.values(), *governed_subissues}
    }
    common = {
        "status": "Todo",
        "moscow": "Must",
        "record type": "Current track",
        "lifecycle": "Open",
        "gate": "Local",
        "owner role": "Maintainer",
        "evidence state": "Unverified",
        "contract version": "1.0.0",
        "sync state": "Clean",
    }
    inverse_tracks = {number: track for track, number in TRACK_ISSUES.items()}
    for issue_number, track_id in inverse_tracks.items():
        item = items.get(issue_number)
        if item is None:
            errors.append(f"{track_id}: Project 28 item missing")
            continue
        expected = {
            **common,
            **EXPECTED_PROJECT[issue_number],
            "track id": track_id,
        }
        for field, value in expected.items():
            if item.get(field) != value:
                errors.append(
                    f"{track_id}: Project field {field!r} expected {value!r}, "
                    f"got {item.get(field)!r}"
                )
    subissue_common = {
        "status": "Todo",
        "record type": "Development ledger",
        "lifecycle": "Open",
        "owner role": "Maintainer",
        "evidence state": "Unverified",
        "sync state": "Clean",
    }
    for issue_number, specific in governed_subissues.items():
        track_id = specific["track id"]
        item = items.get(issue_number)
        if item is None:
            errors.append(f"{track_id}: Project item #{issue_number} missing")
            continue
        for field, value in {**subissue_common, **specific}.items():
            if field == "parent issue":
                continue
            if item.get(field) != value:
                errors.append(
                    f"{track_id}: Project item #{issue_number} field "
                    f"{field!r} expected {value!r}, got {item.get(field)!r}"
                )

    view_query = (
        'query { user(login:"edithatogo") { projectV2(number:28) { '
        "views(first:100) { nodes { name layout filter } } } } }"
    )
    view_result = _run_json(
        ["gh", "api", "graphql", "-f", f"query={view_query}"], repo
    )
    project_views = {
        node["name"]: {
            "layout": node["layout"],
            "filter": node.get("filter") or "",
        }
        for node in view_result["data"]["user"]["projectV2"]["views"]["nodes"]
    }
    for view_name, expected in EXPECTED_PROJECT_VIEWS.items():
        observed = project_views.get(view_name)
        if observed is None:
            errors.append(f"Project 28 view missing: {view_name}")
            continue
        for field, value in expected.items():
            if observed.get(field) != value:
                errors.append(
                    f"Project 28 view {view_name!r} {field!r} expected "
                    f"{value!r}, got {observed.get(field)!r}"
                )
    return errors


def main() -> int:
    """Validate local and optionally live programme governance."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--live-github", action="store_true")
    args = parser.parse_args()

    errors = validate_local(args.repo)
    if args.live_github:
        errors.extend(validate_live_github(args.repo))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    mode = "local and live GitHub" if args.live_github else "local"
    print(f"validated Rust-first polyglot programme ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
