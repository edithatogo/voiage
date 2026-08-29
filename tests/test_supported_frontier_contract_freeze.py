"""Contract-freeze tests for the issue #318 frontier umbrella."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
TRACK = ROOT / "conductor/archive/supported_frontier_method_completion_20260723"


def _load(name: str) -> dict[str, object]:
    return json.loads((TRACK / name).read_text(encoding="utf-8"))


def test_contract_freeze_matches_exact_governed_child_inventory() -> None:
    contracts = _load("contract-freeze.json")["contracts"]
    children = _load("child-dispositions.json")["children"]
    assert isinstance(contracts, list)
    assert isinstance(children, list)
    assert [row["issue"] for row in contracts] == [row["issue"] for row in children]
    assert len(contracts) == 18
    assert len({row["canonical_id"] for row in contracts}) == 18


def test_all_children_remain_accepted_without_inferred_exclusions() -> None:
    payload = _load("contract-freeze.json")
    contracts = payload["contracts"]
    assert isinstance(contracts, list)
    assert "No reviewed exclusion has been inferred" in payload["scope_policy"]
    assert all(row["accepted"] is True for row in contracts)
    assert all(row["reviewed_exclusion"] is None for row in contracts)


def test_contract_status_partition_is_exact() -> None:
    contracts = _load("contract-freeze.json")["contracts"]
    assert isinstance(contracts, list)
    by_status = {
        status: {
            row["issue"] for row in contracts if row["classification_status"] == status
        }
        for status in {row["classification_status"] for row in contracts}
    }
    assert by_status == {
        "candidate": {556, 557, 559, 570, 572, 582},
        "candidate-census-checkpoint": {593, 594, 596, 597, 598, 599, 600},
        "frozen-experimental": {558, 560, 571, 595, 619},
    }


def test_every_contract_freezes_units_and_positive_claim_boundary() -> None:
    payload = _load("contract-freeze.json")
    contracts = payload["contracts"]
    assert isinstance(contracts, list)
    assert "do not establish installed execution" in payload["promotion_boundary"]
    for row in contracts:
        assert row["category"]
        assert row["estimand_or_contract"]
        assert row["units_and_conditioning"]
