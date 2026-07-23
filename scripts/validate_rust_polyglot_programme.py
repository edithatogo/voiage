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
    313: {"priority": "P0", "risk level": "High", "review due": "2026-08-31"},
    314: {"priority": "P0", "risk level": "Medium", "review due": "2026-08-15"},
    315: {"priority": "P0", "risk level": "High", "review due": "2026-08-31"},
    316: {"priority": "P0", "risk level": "High", "review due": "2026-08-31"},
    317: {"priority": "P1", "risk level": "High", "review due": "2026-09-30"},
    318: {"priority": "P1", "risk level": "Medium", "review due": "2026-09-30"},
    319: {"priority": "P1", "risk level": "High", "review due": "2026-10-31"},
    320: {"priority": "P1", "risk level": "High", "review due": "2026-10-31"},
    321: {"priority": "P2", "risk level": "Medium", "review due": "2026-10-31"},
    322: {"priority": "P1", "risk level": "High", "review due": "2026-10-31"},
    323: {"priority": "P1", "risk level": "High", "review due": "2026-08-31"},
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


def validate_live_github(repo: Path) -> list[str]:
    """Return live GitHub issue, subissue, and Project 28 validation errors."""
    errors: list[str] = []
    query = (
        'query { repository(owner:"edithatogo",name:"voiage") { '
        "issue(number:313) { number subIssues(first:20) { nodes { number } } } } }"
    )
    result = _run_json(["gh", "api", "graphql", "-f", f"query={query}"], repo)
    subissues = {
        node["number"]
        for node in result["data"]["repository"]["issue"]["subIssues"]["nodes"]
    }
    expected_subissues = set(TRACK_ISSUES.values()) - {PARENT_ISSUE}
    if subissues != expected_subissues:
        errors.append(
            f"native subissues mismatch: expected {sorted(expected_subissues)}, "
            f"got {sorted(subissues)}"
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

    project = _run_json(
        [
            "gh",
            "project",
            "item-list",
            "28",
            "--owner",
            "edithatogo",
            "--limit",
            "200",
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
        and item["content"].get("number") in TRACK_ISSUES.values()
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
