from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import cast

import pytest

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
        registry_entry = next(
            entry
            for entry in families
            if frontier_root / cast("str", entry["path"]) == manifest_path
        )
        fixture_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if registry_entry.get("manifest_kind", "split") == "bundled":
            assert manifest["method_maturity"] == "experimental"
            assert (fixture_root / manifest["request_schema"]).is_file()
            assert (fixture_root / manifest["result_schema"]).is_file()
            assert manifest["fixtures"]
            for fixture in manifest["fixtures"]:
                payload = json.loads((fixture_root / fixture["path"]).read_text())
                assert "request" in payload
                assert {"expected", "result"} & set(payload)
            continue
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


@pytest.mark.parametrize("unsafe_kind", ["absolute", "traversal"])
def test_frontier_paths_must_remain_inside_the_governed_root(
    tmp_path: Path, unsafe_kind: str
) -> None:
    unsafe = (
        str(Path(tmp_path.anchor) / "outside.json")
        if unsafe_kind == "absolute"
        else "../outside.json"
    )
    with pytest.raises(validate_frontier_contract.ValidationError, match="forbidden|escapes"):
        validate_frontier_contract._resolve_contained(tmp_path, unsafe, "fixture")


def test_frontier_registry_rejects_duplicate_names_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_root = tmp_path / "fixtures"
    registry_root.mkdir()
    registry = json.loads(Path("specs/frontier/fixtures/manifest.json").read_text())
    registry["families"].append(dict(registry["families"][0]))
    manifest_path = registry_root / "manifest.json"
    schema_path = registry_root / "manifest.schema.json"
    manifest_path.write_text(json.dumps(registry), encoding="utf-8")
    schema_path.write_text(
        Path("specs/frontier/fixtures/manifest.schema.json").read_text(),
        encoding="utf-8",
    )
    monkeypatch.setattr(validate_frontier_contract, "FRONTIER_ROOT", tmp_path)
    monkeypatch.setattr(validate_frontier_contract, "REGISTRY_MANIFEST", manifest_path)
    monkeypatch.setattr(validate_frontier_contract, "REGISTRY_SCHEMA", schema_path)

    with pytest.raises(validate_frontier_contract.ValidationError, match="duplicate"):
        validate_frontier_contract._validate_registry()


def test_bundled_fixture_payload_is_schema_validated_after_hash_verification(
    tmp_path: Path,
) -> None:
    source = Path("specs/frontier/expected-utility-information-pricing/v1")
    family_root = tmp_path / "v1"
    shutil.copytree(source, family_root)
    fixture_path = family_root / "fixtures/normative/affine-clairvoyant.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    del payload["request"]["schema_version"]
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = family_root / "fixtures/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixtures"][0]["sha256"] = hashlib.sha256(
        fixture_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        validate_frontier_contract.ValidationError, match="schema_version.*required"
    ):
        validate_frontier_contract._validate_bundled_family_manifest(manifest_path)
