"""Tests for Industry Decision Contract Binding Parity (#579)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voiage.binding_dispositions import (
    get_binding_disposition,
    load_industry_decision_binding_dispositions,
    validate_binding_dispositions_manifest,
)
from voiage.exceptions import InputError


def test_manifest_validation() -> None:
    assert validate_binding_dispositions_manifest() is True


def test_load_all_contract_dispositions() -> None:
    contracts = load_industry_decision_binding_dispositions()
    assert len(contracts) >= 4
    assert "decision_problem" in contracts
    assert "decision_cards" in contracts
    assert "enterprise_adapters" in contracts
    assert "domain_templates" in contracts

    # Check DecisionProblem parity
    dp = contracts["decision_problem"]
    assert dp.dispositions["python"].status == "implemented"
    assert dp.dispositions["rust"].status == "implemented"
    assert dp.dispositions["r"].status == "implemented"
    assert dp.dispositions["julia"].status == "implemented"
    assert dp.dispositions["mojo"].status == "upstream_blocked"


def test_get_binding_disposition() -> None:
    py_dp = get_binding_disposition("decision_problem", "python")
    assert py_dp.status == "implemented"
    assert py_dp.symbol == "voiage.schema.DecisionProblem"

    mojo_dp = get_binding_disposition("decision_problem", "mojo")
    assert mojo_dp.status == "upstream_blocked"
    assert "external upstream boundary" in mojo_dp.reason


def test_contract_binding_parity_to_dict() -> None:
    contracts = load_industry_decision_binding_dispositions()
    dp = contracts["decision_problem"]
    data = dp.to_dict()
    assert data["contract_id"] == "decision_problem"
    assert data["dispositions"]["python"]["status"] == "implemented"


def test_error_handling() -> None:
    with pytest.raises(ValueError, match="not found in manifest"):
        get_binding_disposition("non_existent_contract", "python")

    with pytest.raises(ValueError, match="not configured"):
        get_binding_disposition("decision_problem", "non_existent_language")

    non_existent_path = Path("specs/abi/missing.json")
    with pytest.raises(InputError, match="not found"):
        load_industry_decision_binding_dispositions(manifest_path=non_existent_path)

    with pytest.raises(InputError, match="not found"):
        validate_binding_dispositions_manifest(manifest_path=non_existent_path)
