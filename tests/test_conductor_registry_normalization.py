"""Regression contracts for historical Conductor registry normalization."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.normalize_conductor_registry import (
    collect_track_records,
    normalize_repository,
)


def test_normalized_repository_is_idempotent_and_complete() -> None:
    root = Path.cwd()
    records = collect_track_records(root)
    registry = root / "conductor/tracks.md"
    registry_before = registry.read_bytes()
    registry_mtime = registry.stat().st_mtime_ns

    assert len(records) == len({record.track_dir.resolve() for record in records})
    assert {
        path.name for path in (root / "conductor/archive").iterdir() if path.is_dir()
    } <= {record.track_dir.name for record in records}
    assert normalize_repository(root, apply=False) == []
    assert registry.read_bytes() == registry_before
    assert registry.stat().st_mtime_ns == registry_mtime


def test_normalization_preserves_external_and_superseded_boundaries() -> None:
    root = Path.cwd()
    audit = json.loads(
        (root / "conductor/registry-normalization-audit.json").read_text(
            encoding="utf-8"
        )
    )

    assert audit["baseline"]["error_count"] == 223
    assert audit["result"]["error_count"] == 0
    assert audit["result"]["ambiguous_track_count"] == 0
    assert audit["policy"]["unchecked_completed_track_items"] == (
        "preserved_as_non_acceptance_follow_up_prose"
    )
    assert audit["policy"]["superseded_status"] == (
        "metadata_completed_with_legacy_outcome_preserved"
    )
    assert audit["policy"]["external_outcomes"] == "never_promoted_by_normalization"

    for record in collect_track_records(root):
        metadata = json.loads(
            (record.track_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["track_id"] == record.track_dir.name
        assert metadata["status"] == record.expected_status


def test_every_track_has_current_required_artifacts_and_links() -> None:
    for record in collect_track_records(Path.cwd()):
        for name in ("index.md", "spec.md", "plan.md", "metadata.json"):
            path = record.track_dir / name
            assert path.is_file()
            assert path.read_text(encoding="utf-8").strip()

        index = (record.track_dir / "index.md").read_text(encoding="utf-8")
        assert "[Specification](./spec.md)" in index
        assert "[Implementation Plan](./plan.md)" in index
        assert "[Metadata](./metadata.json)" in index
