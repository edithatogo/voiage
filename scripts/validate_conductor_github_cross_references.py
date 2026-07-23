#!/usr/bin/env python3
"""Validate complete Conductor-to-GitHub traceability."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

MANIFEST_PATH = Path("conductor/github-cross-references.json")
PROJECT_URL = "https://github.com/users/edithatogo/projects/28"


def _local_tracks(root: Path) -> set[str]:
    tracks: set[str] = set()
    for relative in ("conductor/tracks", "conductor/archive"):
        base = root / relative
        tracks.update(path.name for path in base.iterdir() if path.is_dir())
    return tracks


def validate(root: Path) -> list[str]:
    """Return cross-reference validation errors for *root*."""
    errors: list[str] = []
    manifest_file = root / MANIFEST_PATH
    try:
        manifest: dict[str, Any] = json.loads(manifest_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load {MANIFEST_PATH}: {exc}"]

    if manifest.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if manifest.get("project_url") != PROJECT_URL:
        errors.append(f"project_url must be {PROJECT_URL}")

    entries = manifest.get("tracks")
    if not isinstance(entries, list):
        return [*errors, "tracks must be a list"]

    ids: list[str] = []
    issue_urls: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("every track entry must be an object")
            continue
        track_id = entry.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            errors.append("every track entry must have a non-empty track_id")
            continue
        ids.append(track_id)
        issue = entry.get("issue")
        if not isinstance(issue, dict) or not issue.get("url"):
            errors.append(f"{track_id}: missing issue URL")
            continue
        issue_urls.append(str(issue["url"]))
        if not entry.get("parent_issue_url") and not entry.get("subissues"):
            errors.append(f"{track_id}: missing parent issue or owned subissues")

        lifecycle = entry.get("lifecycle")
        if lifecycle == "completed" and issue.get("state") != "closed":
            errors.append(f"{track_id}: completed track issue must be closed")
        if lifecycle == "proposed":
            source_pr = entry.get("source_pull_request")
            if not isinstance(source_pr, dict) or not source_pr.get("url"):
                errors.append(f"{track_id}: proposed track must record source PR")

        prs = entry.get("pull_requests")
        if not isinstance(prs, list):
            errors.append(f"{track_id}: pull_requests must be a list")
        elif (
            lifecycle == "completed"
            and not prs
            and entry.get("pull_request_evidence") != "none_found"
        ):
            errors.append(f"{track_id}: empty PR list must declare none_found evidence")
        for pull_request in prs or []:
            if not isinstance(pull_request, dict) or not pull_request.get("url"):
                errors.append(f"{track_id}: invalid pull-request entry")
            elif lifecycle == "completed" and pull_request.get("status") != "merged":
                errors.append(f"{track_id}: completed-track PR must be merged")

        local_path = root / str(entry.get("path", ""))
        if local_path.is_dir():
            metadata_path = local_path / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                except json.JSONDecodeError as exc:
                    errors.append(f"{track_id}: invalid metadata JSON: {exc}")
                else:
                    cross_reference = metadata.get("github_cross_reference", {})
                    if cross_reference.get("issue") != issue.get("url"):
                        errors.append(f"{track_id}: metadata issue backlink drift")
            index_path = local_path / "index.md"
            if index_path.exists() and str(issue["url"]) not in index_path.read_text():
                errors.append(f"{track_id}: index issue backlink drift")

    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        errors.append(f"duplicate track IDs: {', '.join(duplicates)}")
    duplicate_issues = sorted(
        {value for value in issue_urls if issue_urls.count(value) > 1}
    )
    if duplicate_issues:
        errors.append(f"reused issue URLs: {', '.join(duplicate_issues)}")

    local = _local_tracks(root)
    represented = set(ids)
    missing = sorted(local - represented)
    if missing:
        errors.append(f"local tracks missing from manifest: {', '.join(missing)}")
    unexpected = sorted(
        track_id
        for track_id in represented - local
        if next(
            entry.get("lifecycle")
            for entry in entries
            if entry.get("track_id") == track_id
        )
        != "proposed"
    )
    if unexpected:
        errors.append(
            "non-proposed manifest tracks missing locally: " + ", ".join(unexpected)
        )
    return errors


def main() -> int:
    """Validate the repository supplied on the command line."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    errors = validate(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Conductor GitHub cross-references valid: {root / MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
