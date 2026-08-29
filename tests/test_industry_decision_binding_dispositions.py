"""Tests for Industry Decision Contract Binding Parity (#579)."""

from __future__ import annotations

import json
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
    assert dp.dispositions["rust"].status == "internal"
    assert dp.dispositions["r"].status == "unsupported"
    assert dp.dispositions["julia"].status == "unsupported"
    assert dp.dispositions["mojo"].status == "upstream_blocked"


def test_get_binding_disposition() -> None:
    py_dp = get_binding_disposition("decision_problem", "python")
    assert py_dp.status == "implemented"
    assert py_dp.symbol == "voiage.schema.DecisionProblem"

    mojo_dp = get_binding_disposition("decision_problem", "mojo")
    assert mojo_dp.status == "upstream_blocked"
    assert "external upstream boundary" in mojo_dp.reason


def test_manifest_resolves_implemented_and_contract_only_claims() -> None:
    assert validate_binding_dispositions_manifest(resolve_symbols=True) is True


def test_contract_binding_parity_to_dict() -> None:
    contracts = load_industry_decision_binding_dispositions()
    dp = contracts["decision_problem"]
    data = dp.to_dict()
    assert data["contract_id"] == "decision_problem"
    assert data["dispositions"]["python"]["status"] == "implemented"


def test_error_handling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found in manifest"):
        get_binding_disposition("non_existent_contract", "python")

    with pytest.raises(ValueError, match="not configured"):
        get_binding_disposition("decision_problem", "non_existent_language")

    non_existent_path = Path("specs/abi/missing.json")
    with pytest.raises(InputError, match="not found"):
        load_industry_decision_binding_dispositions(manifest_path=non_existent_path)

    with pytest.raises(InputError, match="not found"):
        validate_binding_dispositions_manifest(manifest_path=non_existent_path)

    # Test empty contracts manifest
    empty_manifest = tmp_path / "empty_manifest.json"
    empty_manifest.write_text(json.dumps({"contracts": {}}))
    with pytest.raises(InputError, match="contains no contracts"):
        validate_binding_dispositions_manifest(manifest_path=empty_manifest)

    # Test unknown language
    bad_lang_manifest = tmp_path / "bad_lang.json"
    bad_lang_manifest.write_text(
        json.dumps(
            {"contracts": {"c1": {"dispositions": {"ruby": {"status": "implemented"}}}}}
        )
    )
    with pytest.raises(InputError, match="Unknown language"):
        validate_binding_dispositions_manifest(manifest_path=bad_lang_manifest)

    # Test invalid status
    bad_status_manifest = tmp_path / "bad_status.json"
    bad_status_manifest.write_text(
        json.dumps(
            {
                "contracts": {
                    "c1": {"dispositions": {"python": {"status": "invalid_status_xyz"}}}
                }
            }
        )
    )
    with pytest.raises(InputError, match="Invalid status"):
        validate_binding_dispositions_manifest(manifest_path=bad_status_manifest)


@pytest.mark.parametrize(
    ("language", "disposition", "schema", "message"),
    [
        ("r", {"status": "unsupported"}, None, "Missing reason"),
        (
            "python",
            {"status": "implemented", "symbol": "invalid"},
            None,
            "Invalid Python symbol",
        ),
        (
            "python",
            {"status": "implemented", "symbol": "voiage.schema.MissingThing"},
            None,
            "Unresolvable Python symbol",
        ),
        (
            "r",
            {"status": "contract_only", "symbol": "specs/missing.schema.json"},
            None,
            "Unresolvable contract",
        ),
        ("r", {"status": "adapter"}, "specs/missing.schema.json", "Missing schema"),
    ],
)
def test_symbol_resolution_rejects_each_invalid_disposition(
    tmp_path: Path,
    language: str,
    disposition: dict[str, str],
    schema: str | None,
    message: str,
) -> None:
    manifest = tmp_path / "invalid-disposition.json"
    manifest.write_text(
        json.dumps(
            {
                "contracts": {
                    "example": {
                        "schema": schema
                        or "specs/core-api/schemas/v1/decision-problem.schema.json",
                        "dispositions": {language: disposition},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match=message):
        validate_binding_dispositions_manifest(
            manifest_path=manifest,
            resolve_symbols=True,
        )
