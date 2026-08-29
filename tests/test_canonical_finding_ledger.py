"""Contracts for the canonical pre-submission finding ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
LEDGER_PATH = (
    ROOT / "specs" / "submission-readiness" / "canonical-finding-ledger-20260829.json"
)


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_source_register_is_complete_and_hash_bound() -> None:
    ledger = _ledger()
    sources = ledger["source_register"]

    assert len(sources) == 7
    for source in sources:
        path = ROOT / source["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_every_audit_finding_has_exactly_one_canonical_disposition() -> None:
    ledger = _ledger()
    findings = ledger["findings"]
    canonical_ids = [finding["id"] for finding in findings]
    source_ids: set[str] = set()

    for source in ledger["source_register"]:
        path = ROOT / source["path"]
        if path.suffix != ".json" or "scalene" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_ids.update(finding["id"] for finding in payload.get("findings", []))

    assert len(canonical_ids) == len(set(canonical_ids)) == 34
    assert source_ids == set(canonical_ids) - {
        "ANALYTIC-001",
        "ANALYTIC-002",
        "ANALYTIC-003",
        "ANALYTIC-004",
    }
    assert set(canonical_ids) >= {f"ANALYTIC-{number:03d}" for number in range(1, 5)}


def test_every_finding_has_owner_validation_and_submission_impact() -> None:
    ledger = _ledger()
    allowed = set(ledger["allowed_dispositions"])

    for finding in ledger["findings"]:
        assert finding["disposition"] in allowed
        assert finding["owner_boundary"] in {
            "repository",
            "upstream",
            "human",
            "external",
        }
        assert finding["validation_path"]
        assert all(
            re.fullmatch(r"P[2-7]-T\d+", task) for task in finding["validation_path"]
        )
        assert finding["submission_impact"]


def test_migrated_track_coverage_matches_the_frozen_manifest() -> None:
    ledger = _ledger()
    migrated = ledger["migrated_track_coverage"]
    manifest = (
        ROOT / "conductor" / "archive" / ledger["track_id"] / "migration-manifest.md"
    ).read_text(encoding="utf-8")
    manifest_rows = re.findall(
        r"\| `([^`]+)` \| (\d+) \| `([^`]+)` \| `[0-9a-f]{64}` \|",
        manifest,
    )
    expected = {track: (int(count), refs) for track, count, refs in manifest_rows}

    assert len(migrated) == len(expected) == 21
    assert {entry["source_track"] for entry in migrated} == expected.keys()
    for entry in migrated:
        count, refs = expected[entry["source_track"]]
        assert entry["source_refs"] == refs
        assert len(refs.split(",")) == count
        assert entry["primary_disposition"] in ledger["allowed_dispositions"]
        assert set(entry["secondary_dispositions"]) <= set(
            ledger["allowed_dispositions"]
        )
        assert entry["destination_tasks"]
        assert entry["boundary"]

    assert sum(count for count, _refs in expected.values()) == 161


def test_declared_counts_match_the_canonical_records() -> None:
    ledger = _ledger()
    findings = ledger["findings"]
    counts = ledger["counts"]

    assert counts["canonical_findings"] == len(findings)
    assert (
        counts["source_audit_findings"]
        == len(findings) - counts["analytical_candidates"]
    )
    assert counts["migrated_source_tracks"] == len(ledger["migrated_track_coverage"])
    for disposition in (
        "must_fix",
        "accepted_limitation",
        "experimental_or_preview",
        "reviewed_exclusion",
    ):
        assert counts[f"{disposition}_findings"] == sum(
            finding["disposition"] == disposition for finding in findings
        )
