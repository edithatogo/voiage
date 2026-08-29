"""Contract checks for the immutable v2.1.0 publication receipt."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "conductor"
    / "archive"
    / "quality_release_automation_20260723"
    / "release-2.1.0-publication-receipt-20260821.json"
)


def test_release_2_1_0_receipt_binds_immutable_publication() -> None:
    """The receipt must bind the reviewed payload to both public destinations."""
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))

    assert payload["release"]["version"] == "2.1.0"
    assert payload["release"]["tag"] == "v2.1.0"
    assert payload["release"]["commit"] == ("964a0fc334ece9509387cd07d43776adf38be240")
    assert payload["release"]["tag_signature_verified"] is True
    assert payload["release"]["frontier_promoted"] is False
    assert payload["github"]["immutable"] is True
    assert payload["github"]["draft"] is False
    assert payload["pypi"]["latest_version"] == "2.1.0"
    assert payload["external_gates"]["registry_acceptance"] == "pending"
    assert payload["external_gates"]["publication_acceptance"] == "pending"

    expected = payload["reviewed_digests"]
    published = {
        artifact["filename"]: artifact["sha256"]
        for artifact in payload["pypi"]["artifacts"]
    }
    assert published == expected
    assert len([name for name in expected if name.endswith(".whl")]) == 3
    assert len([name for name in expected if name.endswith(".tar.gz")]) == 1
