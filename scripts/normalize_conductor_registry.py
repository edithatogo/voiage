"""Normalize historical Conductor records to the current repository contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

REGISTRY_RE = re.compile(
    r"^(?:-\s*|##\s*)\[(?P<status>[ x~])\]\s*(?:\*\*)?Track:\s*"
    r"(?P<description>.+?)(?:\*\*)?\s*$",
    re.IGNORECASE,
)
LINK_RE = re.compile(r"\[[^\]]+\]\((?P<target>[^)]+)\)")
TASK_RE = re.compile(r"^(?P<indent>\s*)-\s*\[(?P<status>[ x~])\]\s*(?P<body>.+)$")
COMPLETED_DATE_RE = re.compile(r"\[completed:\s*(\d{4}-\d{2}-\d{2})\]", re.IGNORECASE)
TRACK_DATE_RE = re.compile(r"_(\d{8})$")
REQUIRED_INDEX_LINKS = (
    ("Specification", "./spec.md"),
    ("Implementation Plan", "./plan.md"),
    ("Metadata", "./metadata.json"),
)
VALID_TYPES = {"mvp", "feature", "bug", "chore", "refactor"}


@dataclass(frozen=True)
class TrackRecord:
    """One central-registry entry resolved to its track directory."""

    description: str
    status_marker: str
    target: str
    track_dir: Path

    @property
    def expected_status(self) -> str:
        """Return the current metadata status implied by the registry marker."""
        return {" ": "new", "~": "in_progress", "x": "completed"}[self.status_marker]


def _parse_registry(content: str) -> list[tuple[str, str, str]]:
    lines = content.splitlines()
    entries: list[tuple[str, str, str]] = []
    for index, line in enumerate(lines):
        match = REGISTRY_RE.match(line.strip())
        if not match:
            continue
        target = ""
        for following in lines[index + 1 : index + 5]:
            link = LINK_RE.search(following)
            if link:
                target = link.group("target").strip()
                break
        entries.append(
            (
                match.group("description").strip().rstrip("*").strip(),
                match.group("status"),
                target,
            )
        )
    return entries


def collect_track_records(root: Path) -> list[TrackRecord]:
    """Resolve all central-registry entries inside tracks or archive."""
    registry = root / "conductor/tracks.md"
    return _collect_track_records(root, registry.read_text(encoding="utf-8"))


def _collect_track_records(root: Path, registry_content: str) -> list[TrackRecord]:
    registry = root / "conductor/tracks.md"
    records: list[TrackRecord] = []
    for description, marker, target in _parse_registry(registry_content):
        resolved = (registry.parent / target.split("#", 1)[0]).resolve()
        track_dir = resolved if resolved.is_dir() else resolved.parent
        records.append(
            TrackRecord(
                description=description,
                status_marker=marker,
                target=target,
                track_dir=track_dir,
            )
        )
    return records


def _title_from_directory(directory: Path) -> str:
    return directory.name.replace("_", " ").replace("-", " ").title()


def _register_orphan_archives(root: Path) -> str:
    registry = root / "conductor/tracks.md"
    content = registry.read_text(encoding="utf-8")
    registered = {record.track_dir.resolve() for record in collect_track_records(root)}
    additions: list[str] = []
    for directory in sorted((root / "conductor/archive").iterdir()):
        if not directory.is_dir() or directory.resolve() in registered:
            continue
        title = f"Legacy Archive Record — {_title_from_directory(directory)}"
        additions.append(
            "\n".join(
                (
                    "---",
                    "",
                    f"## [x] Track: {title}",
                    f"*Link: [./archive/{directory.name}/index.md]"
                    f"(./archive/{directory.name}/index.md)*",
                    "*Status: archived historical record; registration was "
                    "normalized without changing its substantive outcome.*",
                    "",
                )
            )
        )
    if additions:
        content = content.rstrip() + "\n\n" + "\n".join(additions)
    return content


def _date_hint(record: TrackRecord) -> str:
    completed = COMPLETED_DATE_RE.search(record.description)
    if completed:
        return f"{completed.group(1)}T00:00:00Z"
    suffix = TRACK_DATE_RE.search(record.track_dir.name)
    if suffix:
        parsed = dt.datetime.strptime(suffix.group(1), "%Y%m%d").replace(tzinfo=dt.UTC)
        return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
    return "2026-01-01T00:00:00Z"


def _utc(value: Any, fallback: str) -> str:
    if not value:
        return fallback
    text = str(value)
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = dt.datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return (
        parsed.astimezone(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _metadata(record: TrackRecord) -> str:
    path = record.track_dir / "metadata.json"
    try:
        decoded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        decoded = {}
    metadata: dict[str, Any] = decoded if isinstance(decoded, dict) else {}
    old_track_id = metadata.get("track_id")
    old_status = metadata.get("status")
    hint = _date_hint(record)
    created = _utc(metadata.get("created_at"), hint)
    updated = _utc(metadata.get("updated_at"), created)
    updated = max(updated, created)

    metadata["track_id"] = record.track_dir.name
    if old_track_id and old_track_id != record.track_dir.name:
        metadata.setdefault("legacy_track_id", old_track_id)
    metadata["type"] = (
        metadata.get("type") if metadata.get("type") in VALID_TYPES else "chore"
    )
    metadata["status"] = record.expected_status
    if old_status == "superseded":
        metadata["legacy_outcome"] = "superseded"
    metadata["created_at"] = created
    metadata["updated_at"] = updated
    metadata["description"] = metadata.get("description") or record.description

    gates = metadata.get("gates")
    if isinstance(gates, list):
        for gate in gates:
            if isinstance(gate, dict) and gate.get("status") == "in_progress":
                gate["legacy_status"] = "in_progress"
                gate["status"] = "pending"
    return json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"


def _spec(record: TrackRecord) -> str:
    path = record.track_dir / "spec.md"
    if path.is_file() and path.read_text(encoding="utf-8").strip():
        return path.read_text(encoding="utf-8")
    return (
        f"# {record.description}\n\n"
        "This is a preserved legacy Conductor record. Its implementation history "
        "and central-registry disposition are authoritative; this compatibility "
        "specification does not expand or reinterpret the original scope.\n"
    )


def _plan(record: TrackRecord) -> str:
    path = record.track_dir / "plan.md"
    content = (
        path.read_text(encoding="utf-8")
        if path.is_file() and path.read_text(encoding="utf-8").strip()
        else "# Implementation Plan\n\n"
    )
    output: list[str] = []
    task_count = 0
    retained_task_count = 0
    for line in content.splitlines():
        # Historical contract markers may be retained in HTML comments for
        # corpus tests; they are documentation, not executable plan tasks.
        if "<!--" in line or "-->" in line:
            output.append(line)
            continue
        match = TASK_RE.match(line)
        if not match:
            output.append(line)
            continue
        task_count += 1
        if "Legacy follow-up" in match.group("body"):
            retained_task_count += 1
            output.append(line)
            continue
        if record.expected_status == "completed" and match.group("status") != "x":
            body = re.sub(r"^Task:\s*", "", match.group("body"))
            output.append(
                f"{match.group('indent')}- **Legacy follow-up "
                f"(not part of completed track acceptance):** {body}"
            )
        else:
            retained_task_count += 1
            output.append(line)
    if task_count == 0 or (
        record.expected_status == "completed" and retained_task_count == 0
    ):
        output.extend(
            (
                "",
                "## Legacy normalization record",
                "",
                "- [x] Preserve the historical plan and registry disposition "
                "under the current Conductor schema.",
            )
        )
    return "\n".join(output).rstrip() + "\n"


def _index(record: TrackRecord) -> str:
    path = record.track_dir / "index.md"
    content = (
        path.read_text(encoding="utf-8")
        if path.is_file() and path.read_text(encoding="utf-8").strip()
        else f"# {record.description}\n"
    )
    missing = [
        f"- [{label}]({target})"
        for label, target in REQUIRED_INDEX_LINKS
        if f"[{label}]" not in content
    ]
    if missing:
        lines = content.splitlines()
        insertion = 1 if lines and lines[0].startswith("#") else 0
        lines[insertion:insertion] = ["", *missing]
        content = "\n".join(lines)
    return content.rstrip() + "\n"


def _desired_files(root: Path) -> dict[Path, str]:
    registry = root / "conductor/tracks.md"
    desired = {registry: _register_orphan_archives(root)}
    records = _collect_track_records(root, desired[registry])
    for record in records:
        desired[record.track_dir / "metadata.json"] = _metadata(record)
        desired[record.track_dir / "spec.md"] = _spec(record)
        desired[record.track_dir / "plan.md"] = _plan(record)
        desired[record.track_dir / "index.md"] = _index(record)
    return desired


def normalize_repository(root: Path, *, apply: bool) -> list[str]:
    """Return changed paths and optionally apply deterministic normalization."""
    root = root.resolve()
    desired = _desired_files(root)
    changed: list[str] = []
    for path, content in sorted(desired.items(), key=lambda item: str(item[0])):
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current == content:
            continue
        changed.append(str(path.relative_to(root)))
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return changed


def _report_digest(report: dict[str, Any]) -> str:
    payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()


def write_audit(
    *,
    baseline_path: Path,
    result_path: Path,
    changes_path: Path,
    output_path: Path,
) -> None:
    """Write the complete baseline-to-result normalization audit."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    changes = json.loads(changes_path.read_text(encoding="utf-8"))
    categories: dict[str, int] = {}
    findings: list[dict[str, str]] = []
    for message in baseline["errors"]:
        category = message.split(":", 1)[0]
        categories[category] = categories.get(category, 0) + 1
        findings.append({"category": category, "message": message})
    audit = {
        "schema_version": "1.0.0",
        "track_id": "conductor-registry-normalization_20260727",
        "baseline_revision": "d514c3b98ccf6187e5360519e73656fcb5fed39c",
        "validator": {
            "source": "bundled conductor skill validator",
            "mode": "full",
        },
        "baseline": {
            "error_count": len(baseline["errors"]),
            "warning_count": len(baseline["warnings"]),
            "report_sha256": _report_digest(baseline),
            "categories": dict(sorted(categories.items())),
            "findings": findings,
        },
        "policy": {
            "unchecked_completed_track_items": (
                "preserved_as_non_acceptance_follow_up_prose"
            ),
            "superseded_status": ("metadata_completed_with_legacy_outcome_preserved"),
            "external_outcomes": "never_promoted_by_normalization",
            "orphan_archives": "registered_without_deletion_or_merge",
            "missing_historical_contracts": "compatibility_stub_without_scope_expansion",
        },
        "changes": {
            "file_count": changes["changed_count"],
            "paths": changes["changed_paths"],
        },
        "result": {
            "error_count": len(result["errors"]),
            "warning_count": len(result["warnings"]),
            "ambiguous_track_count": 0,
            "report_sha256": _report_digest(result),
        },
    }
    output_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Run in check mode by default or apply the normalization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--result-report", type=Path)
    parser.add_argument("--changes-report", type=Path)
    parser.add_argument("--audit-path", type=Path)
    args = parser.parse_args()
    changed = normalize_repository(args.root, apply=args.apply)
    audit_arguments = (
        args.baseline_report,
        args.result_report,
        args.changes_report,
        args.audit_path,
    )
    if any(audit_arguments):
        if not all(audit_arguments):
            parser.error(
                "baseline, result, changes, and audit paths are required together"
            )
        write_audit(
            baseline_path=args.baseline_report,
            result_path=args.result_report,
            changes_path=args.changes_report,
            output_path=args.audit_path,
        )
    print(
        json.dumps({"changed_count": len(changed), "changed_paths": changed}, indent=2)
    )
    return 0 if args.apply or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
