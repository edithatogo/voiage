"""Verify the bounded M22-M24 semantics freeze remains exact and experimental."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
FREEZE = ROOT / "conductor/tracks/supported_frontier_method_completion_20260723/c18-m22-m24-semantics-freeze.json"


def test_m22_m24_semantics_freeze_matches_artifacts() -> None:
    payload = json.loads(FREEZE.read_text())
    assert payload["status"] == "experimental-semantics-frozen"
    assert payload["candidate_commit"] == "cca185ee"
    assert [family["issue"] for family in payload["families"]] == [570, 572, 582]
    assert all(family["panel_validation"] == "pass-with-experimental-limitations" for family in payload["families"])
    assert payload["panel_findings"]["blocking_implementation_findings"] == []
    for relative, expected in payload["artifact_sha256"].items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected, relative
