"""Named-consumer witnesses for versioned, once-only portfolio costs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.information_source_portfolio import (
    INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V2,
    INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V2,
    validate_information_source_portfolio_result,
)
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
    assert result["cost_basis"]["sources"]["survey"]["asset_ids"] == ["lab"]
    assert result["cost_basis"]["shared_costs"] == _payload(owned=owned)["shared_costs"]
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


def test_exact_budget_and_reversed_marginals() -> None:
    payload = _payload()
    payload["constraints"]["max_cost"] = 0.8
    result = information_source_portfolio_value(payload).to_contract_dict()
    sequence = next(
        item
        for item in result["evaluated_sequences"]
        if item["source_sequence"] == ["registry", "survey"]
    )
    assert sequence["total_source_cost"] == pytest.approx(0.8)
    assert [
        item["incremental_source_cost"] for item in sequence["conditional_marginals"]
    ] == pytest.approx([0.7, 0.1])


@pytest.mark.parametrize("kind", ["unknown_source_asset", "unknown_owned_asset"])
def test_unknown_asset_references_fail_closed(kind: str) -> None:
    payload = _payload()
    if kind == "unknown_source_asset":
        payload["sources"][0]["asset_ids"] = ["missing"]
    else:
        payload["shared_costs"]["existing_assets"] = ["missing"]
    with pytest.raises(InputError, match="asset"):
        information_source_portfolio_value(payload)


@pytest.mark.parametrize("kind", ["duplicate", "wrong_unit", "nonfinite"])
def test_invalid_asset_declarations_fail_closed(kind: str) -> None:
    payload = _payload()
    assets = payload["shared_costs"]["assets"]
    if kind == "duplicate":
        assets.append(dict(assets[0]))
    elif kind == "wrong_unit":
        assets[0]["cost_unit"] = "USD"
    else:
        assets[0]["cost"] = float("inf")
    with pytest.raises(InputError, match="asset"):
        information_source_portfolio_value(payload)


def test_v2_packaged_schemas_match_executable_contracts() -> None:
    root = FIXTURE.parents[3] / "v2/schemas"
    for name, expected in (
        ("input.schema.json", INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V2),
        ("result.schema.json", INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V2),
    ):
        assert json.loads((root / name).read_text(encoding="utf-8")) == expected


def test_v2_normative_files_validate_against_external_schemas() -> None:
    root = FIXTURE.parents[3] / "v2"
    payload = json.loads(
        (root / "fixtures/normative/input.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (root / "fixtures/normative/expected.json").read_text(encoding="utf-8")
    )
    input_schema = json.loads(
        (root / "schemas/input.schema.json").read_text(encoding="utf-8")
    )
    output_schema = json.loads(
        (root / "schemas/result.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(input_schema).validate(payload)
    result = information_source_portfolio_value(payload).to_contract_dict()
    Draft202012Validator(output_schema).validate(result)
    sequence = next(
        item
        for item in result["evaluated_sequences"]
        if item["source_sequence"] == expected["sequence"]
    )
    assert sequence["total_source_cost"] == pytest.approx(
        expected["unowned_total_cost"]
    )
    assert [
        item["incremental_source_cost"] for item in sequence["conditional_marginals"]
    ] == pytest.approx(expected["unowned_marginal_costs"])
    payload["shared_costs"]["existing_assets"] = ["lab"]
    result = information_source_portfolio_value(payload).to_contract_dict()
    Draft202012Validator(output_schema).validate(result)
    sequence = next(
        item
        for item in result["evaluated_sequences"]
        if item["source_sequence"] == expected["sequence"]
    )
    assert sequence["total_source_cost"] == pytest.approx(expected["owned_total_cost"])
    assert [
        item["incremental_source_cost"] for item in sequence["conditional_marginals"]
    ] == pytest.approx(expected["owned_marginal_costs"])


def test_original_v1_normative_result_is_unchanged() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected = json.loads(
        (FIXTURE.parent / "expected.json").read_text(encoding="utf-8")
    )
    result = information_source_portfolio_value(payload).to_contract_dict()
    assert result["schema_version"] == "1.0.0"
    assert "cost_basis" not in result
    assert result["baseline"]["value"] == pytest.approx(expected["baseline_value"])
    assert result["optimum"]["source_sequence"] == expected["selected_sequence"]
    assert result["optimum"]["gross_value"] == pytest.approx(expected["gross_value"])
    assert result["optimum"]["net_value"] == pytest.approx(expected["net_value"])


def test_v2_result_rejects_tampered_cost_basis() -> None:
    result = information_source_portfolio_value(_payload()).to_contract_dict()
    result["cost_basis"]["sources"]["survey"]["cost"] = 9.0
    with pytest.raises(ValueError, match="cost basis"):
        validate_information_source_portfolio_result(result)


def test_v2_result_rejects_tampered_optimum_cost() -> None:
    result = information_source_portfolio_value(_payload()).to_contract_dict()
    result["optimum"]["total_source_cost"] = 999.0
    with pytest.raises(ValueError, match="optimum"):
        validate_information_source_portfolio_result(result)


@settings(max_examples=30, deadline=None)
@given(
    direct_a=st.integers(min_value=0, max_value=100),
    direct_b=st.integers(min_value=0, max_value=100),
    fixed=st.integers(min_value=0, max_value=100),
    owned=st.booleans(),
)
def test_v2_two_source_union_matches_integer_oracle(
    direct_a: int, direct_b: int, fixed: int, owned: bool
) -> None:
    payload = _payload(owned=owned)
    payload["sources"][0]["cost"] = direct_a
    payload["sources"][1]["cost"] = direct_b
    payload["shared_costs"]["assets"][0]["cost"] = fixed
    payload["constraints"]["max_cost"] = 1000.0
    result = information_source_portfolio_value(payload).to_contract_dict()
    fixed_due = 0 if owned else fixed
    for order, expected_marginals in (
        (["survey", "registry"], [direct_a + fixed_due, direct_b]),
        (["registry", "survey"], [direct_b + fixed_due, direct_a]),
    ):
        sequence = next(
            item
            for item in result["evaluated_sequences"]
            if item["source_sequence"] == order
        )
        assert sequence["total_source_cost"] == direct_a + direct_b + fixed_due
        assert [
            item["incremental_source_cost"]
            for item in sequence["conditional_marginals"]
        ] == expected_marginals


def test_v2_input_order_does_not_change_result() -> None:
    payload = _payload()
    expected = information_source_portfolio_value(payload).to_contract_dict()
    payload["sources"].reverse()
    payload["states"].reverse()
    observed = information_source_portfolio_value(payload).to_contract_dict()
    assert observed == expected
