from __future__ import annotations

import json

import jsonschema

from scripts.validate_vop_compatibility import BASE, validate


def test_pinned_vop_contract_matches_voiage_runtime() -> None:
    contract = validate()
    assert contract["profiles"]["directional_regret"]["producer"] == "voiage"


def test_pinned_vop_contract_matches_strict_schema() -> None:
    schema = json.loads((BASE / "compatibility-contract.schema.json").read_text())
    contract = json.loads((BASE / "v1/contract.json").read_text())
    jsonschema.validate(contract, schema)
    assert schema["additionalProperties"] is False
