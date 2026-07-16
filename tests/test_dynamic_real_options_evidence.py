from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
    / "specs"
    / "frontier"
    / "dynamic-real-options"
    / "v1"
)


def test_dynamic_real_options_evidence_is_hash_pinned_and_scoped() -> None:
    evidence = json.loads((ROOT / "fixtures" / "evidence.json").read_text())
    assert evidence["maturity"] == "fixture-backed"
    assert evidence["stable_claim_allowed"] is False
    assert evidence["open_data"]["status"] == "blocked_external"
    assert evidence["open_data"]["next_action"]
    assert evidence["parity"]["python"] == "verified"
    assert evidence["parity"]["rust"] == "deferred_with_rationale"
    assert len(evidence["artifacts"]) == 5
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
