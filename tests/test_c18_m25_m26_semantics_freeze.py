"""Verify the bounded M25-M26 semantics freeze remains exact and experimental."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
FREEZE = ROOT / "conductor/tracks/supported_frontier_method_completion_20260723/c18-m25-m26-semantics-freeze.json"


def test_m25_m26_semantics_freeze_matches_artifacts() -> None:
    payload = json.loads(FREEZE.read_text())
    assert payload["status"] == "experimental-semantics-frozen"
    assert payload["candidate_commit"] == "291347ba"
    assert [family["issue"] for family in payload["families"]] == [593, 594]
    assert all(family["panel_validation"] == "pass-with-experimental-limitations" for family in payload["families"])
    assert payload["panel_findings"]["blocking_implementation_findings"] == []
    for relative, expected in payload["artifact_sha256"].items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected, relative
