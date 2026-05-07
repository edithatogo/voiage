from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from scripts import validate_frontier_contract


def test_frontier_fixture_manifests_and_artifacts_are_consistent() -> None:
    """Frontier fixture manifests should point at real deterministic artifacts."""
    frontier_root = Path("specs/frontier")
    registry_root = frontier_root / "fixtures"
    registry = json.loads((registry_root / "manifest.json").read_text())
    assert registry["version"] == "v1"
    assert registry["status"] == "registry"
    families = cast("list[dict[str, object]]", registry["families"])
    assert families

    schema = json.loads((registry_root / "manifest.schema.json").read_text())
    assert schema["title"] == "FrontierFixtureRegistryV1"

    manifests = [frontier_root / cast("str", entry["path"]) for entry in families]
    assert manifests, "expected at least one frontier fixture manifest"

    for manifest_path in manifests:
        fixture_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["version"] == "v1"
        assert manifest["status"] == "fixture-backed"

        normative = cast("list[dict[str, object]]", manifest["normative"])
        assert normative, (
            f"{manifest_path} should define at least one normative fixture"
        )
        for entry in normative:
            input_artifact = cast("str", entry["input_artifact"])
            output_artifact = cast("str", entry["expected_output_artifact"])
            assert (fixture_root / input_artifact).is_file()
            assert (fixture_root / output_artifact).is_file()


def test_validate_frontier_contract_entrypoint_returns_zero() -> None:
    """The standalone frontier contract validator should succeed."""
    assert validate_frontier_contract.main() == 0
