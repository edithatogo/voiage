#!/usr/bin/env python3
"""Validate the frontier VOI fixture registry and family manifests."""

from __future__ import annotations

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
        validated.append({"name": name, "path": relpath})
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


def main() -> int:
    """Validate all frontier fixture registry entries."""
    family_entries = _validate_registry()
    for entry in family_entries:
        manifest_path = FRONTIER_ROOT / cast("str", entry["path"])
        if not manifest_path.is_file():
            raise ValidationError(f"missing frontier family manifest: {manifest_path}")
        _validate_family_manifest(manifest_path)

    print(f"validated {REGISTRY_MANIFEST.relative_to(REPO_ROOT)}")
    for entry in family_entries:
        print(f"validated {entry['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
