"""Keep installed runtime contracts synchronized with the source distribution."""

from pathlib import Path

lazy import pytest

import voiage
from voiage import _contract_root


def test_private_contracts_take_precedence_over_neighboring_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "voiage"
    (package / "_contracts" / "specs").mkdir(parents=True)
    (tmp_path / "specs").mkdir()
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(_contract_root, "__file__", str(package / "_contract_root.py"))
    assert _contract_root.contract_root() == package / "_contracts"


def test_runtime_contract_json_matches_source() -> None:
    """An installed wheel must expose every canonical JSON contract unchanged."""
    source = Path(__file__).resolve().parents[2] / "specs"
    package = Path(voiage.__file__).resolve().parent
    installed = (
        source if package.parent == source.parent else package / "_contracts" / "specs"
    )
    contracts = sorted(source.rglob("*.json"))
    assert contracts, "No source contracts were found"
    for contract in contracts:
        relative = contract.relative_to(source)
        packaged = installed / relative
        assert packaged.is_file(), f"Missing installed contract: {relative}"
        assert packaged.read_bytes() == contract.read_bytes(), relative
