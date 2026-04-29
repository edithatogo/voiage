#!/usr/bin/env python3
"""Validate the core API spec examples against the local v1 JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = REPO_ROOT / "specs" / "core-api"
SCHEMA_ROOT = SPEC_ROOT / "schemas" / "v1"
EXAMPLE_ROOT = SPEC_ROOT / "examples" / "v1"
FIXTURE_ROOT = SPEC_ROOT / "fixtures" / "v1"
FIXTURE_MANIFEST = FIXTURE_ROOT / "manifest.json"


class ValidationError(Exception):
    """Raised when a schema or example violates the contract."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{path}: expected non-empty string")
    return value


def _resolve_ref(ref: str, base_path: Path) -> Path:
    if ref.startswith("./"):
        return (base_path.parent / ref).resolve()
    raise ValidationError(f"Unsupported schema reference: {ref}")


def _validate_scalar(value: Any, schema: dict[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            raise ValidationError(f"{path}: expected string")
        if schema.get("minLength") is not None and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: string is shorter than minLength")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError(f"{path}: expected number")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(f"{path}: expected integer")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValidationError(f"{path}: expected boolean")

    if "const" in schema and value != schema["const"]:
        raise ValidationError(f"{path}: expected constant {schema['const']!r}")

    if schema.get("minimum") is not None and value < schema["minimum"]:
        raise ValidationError(f"{path}: value is below minimum")
    if schema.get("exclusiveMinimum") is not None and value <= schema["exclusiveMinimum"]:
        raise ValidationError(f"{path}: value is not above exclusiveMinimum")


def _validate(value: Any, schema: dict[str, Any], path: str, schema_path: Path) -> None:
    if "$ref" in schema:
        ref_schema = _load_json(_resolve_ref(schema["$ref"], schema_path))
        _validate(value, ref_schema, path, _resolve_ref(schema["$ref"], schema_path))
        return

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValidationError(f"{path}: expected object")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValidationError(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}", schema_path)
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                raise ValidationError(f"{path}: unexpected property {key!r}")
            if isinstance(additional, dict):
                _validate(item, additional, f"{path}.{key}", schema_path)
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValidationError(f"{path}: expected array")
        if schema.get("minItems") is not None and len(value) < schema["minItems"]:
            raise ValidationError(f"{path}: array is shorter than minItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{index}]", schema_path)
    else:
        _validate_scalar(value, schema, path)


def _validate_example(example_path: Path, schema_path: Path) -> None:
    example = _load_json(example_path)
    schema = _load_json(schema_path)
    _validate(example, schema, "$", schema_path)


def _resolve_fixture_artifact(ref: str) -> Path:
    artifact_path = (FIXTURE_ROOT / ref).resolve()
    try:
        artifact_path.relative_to(FIXTURE_ROOT.resolve())
    except ValueError as exc:
        raise ValidationError(f"artifact escapes fixture root: {ref}") from exc
    return artifact_path


def _validate_fixture_manifest_entry(entry: dict[str, Any], section: str, index: int) -> None:
    entry_path = f"$.{section}[{index}]"
    _require_non_empty_string(entry.get("name"), f"{entry_path}.name")
    _require_non_empty_string(entry.get("method_family"), f"{entry_path}.method_family")
    artifact = _require_non_empty_string(
        entry.get("expected_output_artifact"),
        f"{entry_path}.expected_output_artifact",
    )
    _require_non_empty_string(entry.get("tolerance_policy"), f"{entry_path}.tolerance_policy")
    provenance = entry.get("provenance")
    if section == "normative":
        if not isinstance(provenance, dict):
            raise ValidationError(f"{entry_path}.provenance: expected object")
        seed = provenance.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValidationError(f"{entry_path}.provenance.seed: expected integer")
        execution_mode = provenance.get("execution_mode")
        if execution_mode != "deterministic":
            raise ValidationError(
                f"{entry_path}.provenance.execution_mode: expected 'deterministic'"
            )
    elif provenance is not None:
        raise ValidationError(f"{entry_path}.provenance: only normative fixtures may declare provenance")

    artifact_path = _resolve_fixture_artifact(artifact)
    if not artifact_path.is_file():
        raise ValidationError(f"{entry_path}.expected_output_artifact: missing artifact {artifact}")


def _validate_fixture_manifest(manifest_path: Path = FIXTURE_MANIFEST) -> None:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValidationError("fixture manifest must be an object")
    if manifest.get("version") != "v1":
        raise ValidationError("fixture manifest version must be 'v1'")

    for section in ("normative", "illustrative"):
        value = manifest.get(section)
        if not isinstance(value, list):
            raise ValidationError(f"fixture manifest missing {section} array")

        seen_names: set[str] = set()
        seen_artifacts: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValidationError(f"$.{section}[{index}]: expected object")
            name = _require_non_empty_string(item.get("name"), f"$.{section}[{index}].name")
            artifact = _require_non_empty_string(
                item.get("expected_output_artifact"),
                f"$.{section}[{index}].expected_output_artifact",
            )
            if name in seen_names:
                raise ValidationError(f"$.{section}[{index}].name: duplicate entry {name!r}")
            if artifact in seen_artifacts:
                raise ValidationError(
                    f"$.{section}[{index}].expected_output_artifact: duplicate entry {artifact!r}"
                )
            seen_names.add(name)
            seen_artifacts.add(artifact)
            _validate_fixture_manifest_entry(item, section, index)


def main() -> int:
    _validate_fixture_manifest()
    print(f"validated {FIXTURE_MANIFEST.relative_to(REPO_ROOT)}")

    checks = [
        (SCHEMA_ROOT / "decision-problem.schema.json", EXAMPLE_ROOT / "decision-problem.example.json"),
        (SCHEMA_ROOT / "intervention.schema.json", EXAMPLE_ROOT / "intervention.example.json"),
        (SCHEMA_ROOT / "trial-design.schema.json", EXAMPLE_ROOT / "trial-design.example.json"),
        (SCHEMA_ROOT / "parameter-set.schema.json", EXAMPLE_ROOT / "parameter-set.example.json"),
        (SCHEMA_ROOT / "value-array.schema.json", EXAMPLE_ROOT / "value-array.example.json"),
        (SCHEMA_ROOT / "results" / "evpi.schema.json", EXAMPLE_ROOT / "evpi.example.json"),
        (SCHEMA_ROOT / "results" / "evppi.schema.json", EXAMPLE_ROOT / "evppi.example.json"),
        (SCHEMA_ROOT / "results" / "evsi.schema.json", EXAMPLE_ROOT / "evsi.example.json"),
        (SCHEMA_ROOT / "results" / "enbs.schema.json", EXAMPLE_ROOT / "enbs.example.json"),
        (SCHEMA_ROOT / "results" / "ceac.schema.json", EXAMPLE_ROOT / "ceac.example.json"),
    ]

    for schema_path, example_path in checks:
        _validate_example(example_path, schema_path)
        print(f"validated {example_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
