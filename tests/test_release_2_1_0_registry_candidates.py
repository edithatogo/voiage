"""Contract checks for the v2.1.0 external-registry candidate receipt."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "conductor"
    / "archive"
    / "research_software_registry_readiness_20260721"
    / "release-2.1.0-registry-candidate-receipt-20260821.json"
)


def test_registry_receipt_binds_exact_green_candidates() -> None:
    """Each candidate must bind an exact head to successful hosted evidence."""
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))

    assert payload["release"]["version"] == "2.1.0"
    assert payload["release"]["commit"] == ("964a0fc334ece9509387cd07d43776adf38be240")

    conda = payload["candidates"]["conda_forge"]
    assert conda["candidate_head"] == ("331e3d9b509c7938ed298b79d2c37153f984f4b1")
    assert {
        conda["hosted_evidence"][platform]
        for platform in ("linux_64", "osx_64", "win_64")
    } == {"success"}

    yggdrasil = payload["candidates"]["yggdrasil"]
    assert yggdrasil["candidate_head"] == ("db38b5200cc4ed741c5cfc682a8465395f687b41")
    assert len(yggdrasil["hosted_evidence"]["platforms"]) == 7
    assert set(yggdrasil["hosted_evidence"]["platforms"].values()) == {"success"}


def test_registry_receipt_keeps_external_gates_fail_closed() -> None:
    """Green candidates are not registry acceptance or stable-frontier evidence."""
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))

    assert all(
        candidate["state"] == "open" for candidate in payload["candidates"].values()
    )
    assert all(
        candidate["acceptance"].startswith("pending_external_")
        for candidate in payload["candidates"].values()
    )
    assert payload["boundaries"] == {
        "repository_candidate_preparation": "complete",
        "registry_acceptance": "pending_external",
        "publication_acceptance": "separate_external_gate",
        "frontier_maturity": "experimental",
    }
