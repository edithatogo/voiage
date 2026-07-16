from __future__ import annotations

import hashlib
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
        assert manifest["open_data"]["status"] == "verified_repository_example"
        assert manifest["open_data"]["next_action"]
        assert len(manifest["open_data"]["artifacts"]) == 2
        for artifact in manifest["open_data"]["artifacts"]:
            assert artifact["path"].startswith("fixtures/open-data/")
            assert len(artifact["sha256"]) == 64
            artifact_path = ROOT / family / "v1" / artifact["path"]
            assert artifact_path.is_file()
            assert (
                hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                == artifact["sha256"]
            )
        provenance = json.loads(
            (
                ROOT / family / "v1" / "fixtures" / "open-data" / "provenance.json"
            ).read_text()
        )
        assert provenance["data_url"].startswith("https://")
        assert provenance["license"]
        assert provenance["preprocessing"]
        assert provenance["limitations"]
        assert manifest["parity"]["python"] == "verified"
        assert all(item["sha256"] for item in manifest["artifacts"])
