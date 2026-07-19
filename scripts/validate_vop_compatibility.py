"""Validate VOIAGE's pinned mirror of the VOP compatibility contract."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "specs/integration/vop-voiage/compatibility"


def validate() -> dict[str, object]:
    contract_bytes = (BASE / "v1/contract.json").read_bytes()
    schema_bytes = (BASE / "compatibility-contract.schema.json").read_bytes()
    contract = json.loads(contract_bytes)
    upstream = json.loads((BASE / "UPSTREAM.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert len(upstream["canonical_git_commit"]) == 40
    assert hashlib.sha256(contract_bytes).hexdigest() == upstream["contract_sha256"]
    assert hashlib.sha256(schema_bytes).hexdigest() == upstream["schema_sha256"]
    assert contract["contract_version"] == upstream["contract_version"]
    assert project["requires-python"] == contract["runtime"]["python"]
    declared = project["dependencies"]
    for package, required in contract["shared_dependencies"].items():
        assert any(item.lower().startswith(package) and required in item for item in declared), package
    return contract


if __name__ == "__main__":
    validated = validate()
    print(f"validated pinned VOP compatibility contract {validated['contract_version']}")
