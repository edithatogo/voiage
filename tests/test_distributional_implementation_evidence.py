from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "specs" / "frontier"


def test_distributional_and_implementation_evidence_is_explicit() -> None:
    for family in ("distributional", "implementation"):
        manifest = json.loads(
            (ROOT / family / "v1" / "fixtures" / "evidence.json").read_text()
        )
        assert manifest["maturity"] == "fixture-backed"
        assert manifest["stable_claim_allowed"] is False
        assert manifest["open_data"]["status"] == "blocked_external"
        assert manifest["open_data"]["next_action"]
        assert manifest["parity"]["python"] == "verified"
        assert all(item["sha256"] for item in manifest["artifacts"])
