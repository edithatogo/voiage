"""Contracts for the repository-hardened v2.2.0 release candidate."""

from __future__ import annotations

import json
from pathlib import Path

from voiage.versioning import validate_release_tag, validate_version_sync

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_v2_2_0_is_the_synchronized_release_identity() -> None:
    """Every current package surface must project the selected minor version."""
    canonical, mismatches = validate_version_sync(ROOT)

    assert canonical == "2.2.0"
    assert mismatches == []
    assert validate_release_tag("v2.2.0", ROOT) == "2.2.0"


def test_public_software_metadata_selects_v2_2_0() -> None:
    """Citation and CodeMeta records must identify the candidate consistently."""
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    codemeta = _json("codemeta.json")

    assert "version: 2.2.0" in citation
    assert codemeta["version"] == "2.2.0"
    assert codemeta["releaseNotes"].endswith("/releases/tag/v2.2.0")


def test_submission_packets_target_v2_2_0_without_claiming_publication() -> None:
    """Venue staging follows the candidate while external actions remain false."""
    staging = _json("specs/submission-readiness/pyopensci-submission-staging.json")
    candidate = _json("specs/submission-readiness/pyopensci-submission-candidate.json")
    draft = (ROOT / "docs/release/pyopensci-submission-draft.md").read_text(
        encoding="utf-8"
    )
    ropensci = (
        ROOT / "docs/release/ropensci-presubmission-inquiry-draft.md"
    ).read_text(encoding="utf-8")

    assert staging["candidate_version"] == "2.2.0"
    assert candidate["recommended_candidate"]["version"] == "2.2.0"
    assert all(value is False for value in staging["external_actions"].values())
    assert "Version submitted: 2.2.0" in draft
    assert "Package: `voiageR` 2.2.0" in ropensci


def test_canonical_manuscript_identifies_v2_2_0_candidate() -> None:
    """The LaTeX source and metadata must no longer describe v1.0.0."""
    metadata = _json("paper/metadata.json")
    manuscript = (ROOT / "paper/main.tex").read_text(encoding="utf-8")

    assert "software version 2.2.0" in metadata["comments"]
    assert "Version 2.2.0" in metadata["abstract"]
    assert "Version 2.2.0" in manuscript
