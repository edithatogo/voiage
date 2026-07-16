from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "specs" / "frontier" / "perspective" / "v1"


def test_perspective_catalog_and_promotion_gates_are_explicit() -> None:
    catalog = json.loads((ROOT / "fixtures" / "perspective-catalog.json").read_text())
    evidence = json.loads((ROOT / "fixtures" / "evidence.json").read_text())

    assert catalog["method_maturity"] == "fixture-backed"
    assert catalog["stable_claim_allowed"] is False
    assert [item["id"] for item in catalog["perspectives"]] == [
        "payer",
        "societal",
        "patient",
        "provider",
        "regulator",
        "equity-weighted",
        "custom",
    ]
    assert catalog["assumptions"]["objective_uncertainty"]
    assert evidence["open_data"]["status"] == "blocked_external"
    assert evidence["open_data"]["next_action"]
    assert evidence["parity"]["python"] == "verified"
    assert evidence["stable_claim_allowed"] is False

    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        if artifact["sha256"] != "pending":
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
