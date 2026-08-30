"""Prevent version and human-evidence drift across the two manuscript sources."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_joss_describes_actual_native_implementation_boundaries() -> None:
    paper = " ".join((ROOT / "paper.md").read_text().split())
    assert (
        "R bundles a separate, dependency-free Rust kernel for EVPI and ENBS" in paper
    )
    assert "Python retains the decision record" in paper
    assert "signed release 2.2.0" in paper
    assert "snapshot preserves v2.0.0, not v2.2.0" in paper
    assert (
        "One Rust implementation calculates EVPI for Python, R, and Julia" not in paper
    )


def test_preprint_current_release_and_historical_evidence_are_distinct() -> None:
    for name in ("summary", "conclusion"):
        text = (ROOT / f"paper/sections/{name}.tex").read_text()
        assert "Version 2.2.0" in text
        assert "Version 1.0.0" not in text
    limitations = " ".join(
        (ROOT / "paper/sections/limitations.tex").read_text().split()
    )
    assert "native EVPI and ENBS" in limitations
    assert "only EVPI" not in limitations
    availability = (ROOT / "paper/sections/availability.tex").read_text()
    assert "releases/tag/v2.2.0" in availability
    assert "historical" in availability.lower()


def test_new_ai_assisted_manuscript_does_not_reuse_historical_human_attestation() -> (
    None
):
    assurance = json.loads((ROOT / "paper/joss-editorial-assurance.json").read_text())
    gate = "all_retained_ai_outputs_reviewed_modified_and_validated"
    assert assurance["human_review"][gate] == "pending_explicit_final_confirmation"
    assert assurance["author_attestations"][gate]["confirmed_on"] == "2026-07-27"
