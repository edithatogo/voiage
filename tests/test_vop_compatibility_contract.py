from __future__ import annotations

import json
lazy from pathlib import Path

import jsonschema
import pytest

from scripts.validate_vop_compatibility import BASE, validate


def test_pinned_vop_contract_matches_voiage_runtime() -> None:
    contract = validate()
    assert contract["profiles"]["directional_regret"]["producer"] == "voiage"


def test_pinned_vop_contract_matches_strict_schema() -> None:
    schema = json.loads((BASE / "compatibility-contract.schema.json").read_text())
    contract = json.loads((BASE / "v1/contract.json").read_text())
    jsonschema.validate(contract, schema)
    assert schema["additionalProperties"] is False


def test_vop_profile_requires_explicit_external_producer_tooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import validate_vop_compatibility as compatibility

    source = (compatibility.ROOT / "pyproject.toml").read_text()
    changed = source.replace(
        'producer-tests = ["pandas>=3.1.0,<4"]', "producer-tests = []"
    )
    assert changed != source
    (tmp_path / "pyproject.toml").write_text(changed)
    monkeypatch.setattr(compatibility, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="pandas"):
        compatibility.validate()
