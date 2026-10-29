"""Named-consumer witnesses for versioned, once-only portfolio costs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voiage.exceptions import InputError
from voiage.methods.information_source_portfolio import (
    information_source_portfolio_value,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "specs/frontier/information-source-portfolio/v1/fixtures/normative/input.json"
)


def _payload(*, owned: bool = False) -> dict[str, Any]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["schema_version"] = "2.0.0"
    payload["shared_costs"] = {
        "assets": [{"asset_id": "lab", "cost": 0.5, "cost_unit": "AUD"}],
        "existing_assets": ["lab"] if owned else [],
    }
    for index, source in enumerate(payload["sources"]):
        source["asset_ids"] = ["lab"] if index < 2 else []
    payload["sources"][0]["cost"] = 0.1
    payload["sources"][1]["cost"] = 0.2
    return payload


@pytest.mark.parametrize(
    ("owned", "total", "marginals"),
    [(False, 0.8, [0.6, 0.2]), (True, 0.3, [0.1, 0.2])],
)
def test_shared_asset_is_charged_once_or_offset_when_owned(
    owned: bool, total: float, marginals: list[float]
) -> None:
    result = information_source_portfolio_value(
        _payload(owned=owned)
    ).to_contract_dict()
    assert result["schema_version"] == "2.0.0"
    sequence = next(
        item
        for item in result["evaluated_sequences"]
        if item["source_sequence"] == ["survey", "registry"]
    )
    assert sequence["total_source_cost"] == pytest.approx(total)
    assert [
        m["incremental_source_cost"] for m in sequence["conditional_marginals"]
    ] == pytest.approx(marginals)


def test_budget_uses_union_cost() -> None:
    payload = _payload()
    payload["constraints"]["max_cost"] = 0.75
    result = information_source_portfolio_value(payload).to_contract_dict()
    assert not any(
        set(item["source_sequence"]) == {"survey", "registry"}
        for item in result["evaluated_sequences"]
    )
    assert result["assurance"]["prune_reasons"]["cost"] > 0


@pytest.mark.parametrize("kind", ["unknown_source_asset", "unknown_owned_asset"])
def test_unknown_asset_references_fail_closed(kind: str) -> None:
    payload = _payload()
    if kind == "unknown_source_asset":
        payload["sources"][0]["asset_ids"] = ["missing"]
    else:
        payload["shared_costs"]["existing_assets"] = ["missing"]
    with pytest.raises(InputError, match="asset"):
        information_source_portfolio_value(payload)
