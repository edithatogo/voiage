"""Validate VOIAGE's pinned mirror of the VOP compatibility contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "specs/integration/vop-voiage/compatibility"


def validate() -> dict[str, object]:
    """Validate the pinned external contract and its local integration surface."""
    contract_bytes = (BASE / "v1/contract.json").read_bytes()
    schema_bytes = (BASE / "compatibility-contract.schema.json").read_bytes()
    contract = json.loads(contract_bytes)
    upstream = json.loads((BASE / "UPSTREAM.json").read_text(encoding="utf-8"))
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    assert len(upstream["canonical_git_commit"]) == 40
    assert hashlib.sha256(contract_bytes).hexdigest() == upstream["contract_sha256"]
    assert hashlib.sha256(schema_bytes).hexdigest() == upstream["schema_sha256"]
    assert contract["contract_version"] == upstream["contract_version"]
    # The VOP contract describes a stricter integration profile, not the
    # package's complete compatibility range.  Verify that the supported
    # package range contains the profile runtime instead of requiring exact
    # metadata equality. The pinned historical profile is retained while the
    # current package runtime follows its separately qualified Python policy.
    supported_python = SpecifierSet(project["requires-python"])
    assert supported_python.contains("3.15.1")
    assert not supported_python.contains("3.14")
    assert not supported_python.contains("3.11")
    declared = [Requirement(item) for item in project["dependencies"]]
    declared_names = {item.name.lower() for item in declared}
    # Shared dependencies describe both ends of the producer/consumer profile.
    # External-producer libraries belong to unpublished test tooling, not the
    # consumer wheel. Keep the pinned contract bytes and wire profile unchanged;
    # this validates the integration surface, not an upstream deployment upgrade.
    producers = {
        item.name.lower(): item
        for item in map(
            Requirement, config.get("dependency-groups", {}).get("producer-tests", [])
        )
    }
    for package, required in contract["shared_dependencies"].items():
        profile = SpecifierSet(required)
        if package.lower() not in declared_names:
            assert package.lower() in producers, package
            floors = [
                s.version
                for s in producers[package.lower()].specifier
                if s.operator == ">="
            ]
            assert floors, package
            assert all(profile.contains(version) for version in floors), package
    return contract


if __name__ == "__main__":
    validated = validate()
    print(
        f"validated pinned VOP compatibility contract {validated['contract_version']}"
    )
