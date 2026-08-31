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


def test_citation_metadata_matches_the_verified_public_release() -> None:
    """Citation tooling must project the independently verified publication."""
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    codemeta = _json("codemeta.json")

    assert "version: 2.2.0" in citation
    assert "date-released: 2026-08-30" in citation
    assert "v2.2.0 software release" in citation
    assert codemeta["version"] == "2.2.0"
    assert codemeta["downloadUrl"] == "https://pypi.org/project/voiage/2.2.0/"
    assert codemeta["releaseNotes"].endswith("/releases/tag/v2.2.0")
    assert codemeta["identifier"] == codemeta["releaseNotes"]


def test_submission_packets_target_v2_2_0_without_claiming_submission() -> None:
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
