#!/usr/bin/env python3
"""Validate the frontier VOI fixture registry and family manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTIER_ROOT = REPO_ROOT / "specs" / "frontier"
REGISTRY_ROOT = FRONTIER_ROOT / "fixtures"
REGISTRY_MANIFEST = REGISTRY_ROOT / "manifest.json"
REGISTRY_SCHEMA = REGISTRY_ROOT / "manifest.schema.json"


class ValidationError(Exception):
    """Raised when the frontier contract registry is invalid."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path}: expected non-empty string")
    return value


def _validate_registry() -> list[dict[str, object]]:
    registry = _load_json(REGISTRY_MANIFEST)
    if not isinstance(registry, dict):
        raise ValidationError("frontier registry manifest must be an object")
    if registry.get("version") != "v1":
        raise ValidationError("frontier registry version must be 'v1'")
    if registry.get("status") != "registry":
        raise ValidationError("frontier registry status must be 'registry'")

    schema = _load_json(REGISTRY_SCHEMA)
    if (
        not isinstance(schema, dict)
        or schema.get("title") != "FrontierFixtureRegistryV1"
    ):
        raise ValidationError(
            "frontier registry schema title must be 'FrontierFixtureRegistryV1'"
        )

    families = registry.get("families")
    if not isinstance(families, list) or not families:
        raise ValidationError("frontier registry must define at least one family")

    validated: list[dict[str, object]] = []
    for index, item in enumerate(families):
        if not isinstance(item, dict):
            raise ValidationError(f"$.families[{index}]: expected object")
        name = _require_non_empty_string(item.get("name"), f"$.families[{index}].name")
        relpath = _require_non_empty_string(
            item.get("path"), f"$.families[{index}].path"
        )
        maturity = _require_non_empty_string(
            item.get("method_maturity"), f"$.families[{index}].method_maturity"
        )
        if maturity not in {"experimental", "fixture-backed"}:
            raise ValidationError(
                f"$.families[{index}].method_maturity: expected 'experimental' or 'fixture-backed'"
            )
        manifest_kind = item.get("manifest_kind", "split")
        if manifest_kind not in {"split", "bundled"}:
            raise ValidationError(
                f"$.families[{index}].manifest_kind: expected 'split' or 'bundled'"
            )
        validated.append(
            {"name": name, "path": relpath, "manifest_kind": manifest_kind}
        )
    return validated


def _validate_family_manifest(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValidationError(f"{manifest_path}: manifest must be an object")
    if manifest.get("version") != "v1":
        raise ValidationError(f"{manifest_path}: version must be 'v1'")
    if manifest.get("status") != "fixture-backed":
        raise ValidationError(f"{manifest_path}: status must be 'fixture-backed'")

    normative = manifest.get("normative")
    if not isinstance(normative, list) or not normative:
        raise ValidationError(
            f"{manifest_path}: normative must contain at least one item"
        )

    fixture_root = manifest_path.parent
    for index, entry in enumerate(normative):
        if not isinstance(entry, dict):
            raise ValidationError(
                f"{manifest_path}: normative[{index}] must be an object"
            )
        input_artifact = _require_non_empty_string(
            entry.get("input_artifact"),
            f"{manifest_path}.normative[{index}].input_artifact",
        )
        output_artifact = _require_non_empty_string(
            entry.get("expected_output_artifact"),
            f"{manifest_path}.normative[{index}].expected_output_artifact",
        )
        if not (fixture_root / input_artifact).is_file():
            raise ValidationError(
                f"{manifest_path}: missing input artifact {input_artifact}"
            )
        if not (fixture_root / output_artifact).is_file():
            raise ValidationError(
                f"{manifest_path}: missing output artifact {output_artifact}"
            )


def _validate_bundled_family_manifest(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValidationError(f"{manifest_path}: manifest must be an object")
    schema_version = _require_non_empty_string(
        manifest.get("schema_version"), f"{manifest_path}.schema_version"
    )
    if not schema_version.endswith("-v1"):
        raise ValidationError(f"{manifest_path}: schema_version must end with '-v1'")
    if manifest.get("method_maturity") != "experimental":
        raise ValidationError(
            f"{manifest_path}: bundled manifest must remain experimental"
        )
    fixture_root = manifest_path.parent
    for schema_key in ("request_schema", "result_schema"):
        schema_path = _require_non_empty_string(
            manifest.get(schema_key), f"{manifest_path}.{schema_key}"
        )
        if not (fixture_root / schema_path).is_file():
            raise ValidationError(f"{manifest_path}: missing schema {schema_path}")
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValidationError(f"{manifest_path}: fixtures must not be empty")
    for index, entry in enumerate(fixtures):
        if not isinstance(entry, dict):
            raise ValidationError(f"{manifest_path}: fixtures[{index}] must be an object")
        artifact = _require_non_empty_string(
            entry.get("path"), f"{manifest_path}.fixtures[{index}].path"
        )
        expected_hash = _require_non_empty_string(
            entry.get("sha256"), f"{manifest_path}.fixtures[{index}].sha256"
        )
        artifact_path = fixture_root / artifact
        if not artifact_path.is_file():
            raise ValidationError(f"{manifest_path}: missing bundled artifact {artifact}")
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValidationError(
                f"{manifest_path}: SHA-256 mismatch for bundled artifact {artifact}"
            )
        payload = _load_json(artifact_path)
        if (
            not isinstance(payload, dict)
            or "request" not in payload
            or not ({"expected", "result"} & payload.keys())
        ):
            raise ValidationError(
                f"{manifest_path}: bundled artifact {artifact} must contain "
                "request and expected or result"
            )


def main() -> int:
    """Validate all frontier fixture registry entries."""
    family_entries = _validate_registry()
    for entry in family_entries:
        manifest_path = FRONTIER_ROOT / cast("str", entry["path"])
        if not manifest_path.is_file():
            raise ValidationError(f"missing frontier family manifest: {manifest_path}")
        if entry["manifest_kind"] == "bundled":
            _validate_bundled_family_manifest(manifest_path)
        else:
            _validate_family_manifest(manifest_path)

    print(f"validated {REGISTRY_MANIFEST.relative_to(REPO_ROOT)}")
    for entry in family_entries:
        print(f"validated {entry['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
