"""Tests for contributor CRediT statement and AI contribution transparency (Issue #323)."""

from __future__ import annotations

import json
from pathlib import Path
import yaml

ROOT = Path(__file__).parents[1]
CONTRIBUTORS_MD = ROOT / "CONTRIBUTORS.md"
AI_CONTRIBUTIONS_MD = ROOT / "AI_CONTRIBUTIONS.md"
CITATION_CFF = ROOT / "CITATION.cff"
CODEMETA_JSON = ROOT / "codemeta.json"


def test_contributors_file_exists_and_contains_credit_roles() -> None:
    """CONTRIBUTORS.md must exist and enumerate CRediT taxonomy roles."""
    assert CONTRIBUTORS_MD.exists(), "CONTRIBUTORS.md must exist at root"
    content = CONTRIBUTORS_MD.read_text(encoding="utf-8")

    assert "Contributor Roles Taxonomy (CRediT)" in content
    assert "Conceptualization" in content
    assert "Methodology" in content
    assert "Software" in content
    assert "Formal Analysis" in content
    assert "Investigation" in content
    assert "Writing" in content
    assert "Supervision" in content
    assert "Dylan A Mordaunt" in content
    assert "0000-0002-9775-0603" in content


def test_ai_contributions_file_exists_and_declares_governance() -> None:
    """AI_CONTRIBUTIONS.md must exist, declare non-authorship and human accountability."""
    assert AI_CONTRIBUTIONS_MD.exists(), "AI_CONTRIBUTIONS.md must exist at root"
    content = AI_CONTRIBUTIONS_MD.read_text(encoding="utf-8")

    assert "Non-Authorship" in content
    assert "Human Accountability" in content
    assert "Dylan A Mordaunt" in content
    assert "Transparency Ledger" in content


def test_contributor_metadata_is_synchronized() -> None:
    """Authorship records in CITATION.cff, codemeta.json, and CONTRIBUTORS.md must match."""
    cff_data = yaml.safe_load(CITATION_CFF.read_text(encoding="utf-8"))
    codemeta_data = json.loads(CODEMETA_JSON.read_text(encoding="utf-8"))

    # Validate CITATION.cff
    cff_authors = cff_data["authors"]
    assert len(cff_authors) >= 1
    assert cff_authors[0]["family-names"] == "Mordaunt"
    assert cff_authors[0]["given-names"] == "Dylan"
    assert "0000-0002-9775-0603" in cff_authors[0]["orcid"]

    # Validate codemeta.json
    codemeta_authors = codemeta_data["author"]
    assert len(codemeta_authors) >= 1
    assert codemeta_authors[0]["familyName"] == "Mordaunt"
    assert codemeta_authors[0]["givenName"] == "Dylan"
    assert "0000-0002-9775-0603" in codemeta_authors[0]["@id"]
