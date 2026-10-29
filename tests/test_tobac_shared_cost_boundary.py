"""Show the existing finite-world contract's shared-cost boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voiage.exceptions import InputError
from voiage.methods.information_source_portfolio import (
    information_source_portfolio_value,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "specs/frontier/information-source-portfolio/v1/fixtures/normative/input.json"
)


def _payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_v1_sums_direct_costs_without_a_shared_asset() -> None:
    """Record the established scalar-cost behavior for compatibility."""
    payload = _payload()
    payload["sources"][0]["cost"] = 0.1
    payload["sources"][1]["cost"] = 0.2
    result = information_source_portfolio_value(payload).to_contract_dict()
    sequence = next(
        item
        for item in result["evaluated_sequences"]
        if item["source_sequence"] == ["survey", "registry"]
    )
    assert sequence["total_source_cost"] == pytest.approx(0.3)


def test_v1_rejects_undeclared_shared_asset_data() -> None:
    """Unknown cost identity cannot be silently ignored as a v1 source."""
    payload = _payload()
    payload["sources"][0]["shared_assets"] = ["lab"]
    with pytest.raises(InputError, match="Additional properties are not allowed"):
        information_source_portfolio_value(payload)
