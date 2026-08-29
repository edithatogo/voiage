"""Contracts for the dated pre-submission requirements baseline."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
MANIFEST = (
    ROOT
    / "specs/submission-readiness/current-requirements-source-manifest-20260829.json"
)
BASELINE = ROOT / "specs/submission-readiness/current-requirements-baseline-20260829.md"


def test_source_manifest_is_dated_unique_and_revision_pinned() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed = datetime.fromisoformat(payload["observed_at"])
    assert observed.tzinfo is not None
    assert payload["external_action_performed"] is False
    assert payload["selected_sequence"] == [
        "complete_all_repository_repairs",
        "pyopensci",
        "joss_partner_fast_track",
        "ropensci",
    ]

    sources = payload["sources"]
    ids = [source["id"] for source in sources]
    assert len(ids) == len(set(ids))
    assert all(source["url"].startswith("https://") for source in sources)
    for source in sources:
        if "repository" in source:
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
            revision_date = datetime.fromisoformat(source["revision_observed_at"])
            assert revision_date <= observed
        else:
            assert source["document_version"]


def test_baseline_covers_every_selected_authority_and_preserves_gates() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    text = BASELINE.read_text(encoding="utf-8")
    required_prefixes = {"PYOS", "ROSCI", "JOSS", "PYPA", "SLSA", "OSSF", "FAIR4RS"}
    observed_prefixes = {source["id"].split("-", 1)[0] for source in payload["sources"]}
    assert required_prefixes <= observed_prefixes
    assert "No inquiry, submission" in text
    assert "must not replace\n  `paper/main.tex`" in text
    assert "item-level `@srrstats`" in text
    assert "two-party source\n  review is not assumed" in text
    for source in payload["sources"]:
        assert f"`{source['id']}`" in text
