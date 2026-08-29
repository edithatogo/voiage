"""Verify the bounded M27-M31 semantics freeze remains exact and experimental."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
FREEZE = (
    ROOT
    / "conductor/archive/supported_frontier_method_completion_20260723/c18-m27-m31-semantics-freeze.json"
)


def test_m27_m31_semantics_freeze_matches_artifacts() -> None:
    payload = json.loads(FREEZE.read_text())
    assert payload["status"] == "experimental-semantics-frozen"
    assert payload["candidate_commit"] == "291347ba"
    assert [family["issue"] for family in payload["families"]] == [
        596,
        597,
        598,
        599,
        600,
    ]
    assert all(
        family["panel_validation"] == "pass-with-experimental-limitations"
        for family in payload["families"]
    )
    assert payload["panel_findings"]["blocking_implementation_findings"] == []
    for relative, expected in payload["artifact_sha256"].items():
        path = ROOT / relative
        if not path.exists():
            path = ROOT / relative.replace("conductor/tracks/", "conductor/archive/", 1)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, relative
