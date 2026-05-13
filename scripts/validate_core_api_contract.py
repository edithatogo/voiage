#!/usr/bin/env python3
"""Validate the core API spec examples against the local v1 JSON Schemas."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = REPO_ROOT / "specs" / "core-api"
SCHEMA_ROOT = SPEC_ROOT / "schemas" / "v1"
EXAMPLE_ROOT = SPEC_ROOT / "examples" / "v1"
FIXTURE_ROOT = SPEC_ROOT / "fixtures" / "v1"
FIXTURE_MANIFEST = FIXTURE_ROOT / "manifest.json"
RESULT_SCHEMA_BY_METHOD_FAMILY: dict[str, Path] = {
    "evpi": SCHEMA_ROOT / "results" / "evpi.schema.json",
    "evppi": SCHEMA_ROOT / "results" / "evppi.schema.json",
    "evsi": SCHEMA_ROOT / "results" / "evsi.schema.json",
    "enbs": SCHEMA_ROOT / "results" / "enbs.schema.json",
    "ceac": SCHEMA_ROOT / "results" / "ceac.schema.json",
}


class ValidationError(Exception):
    """Raised when a schema or example violates the contract."""


@dataclass(frozen=True)
class FixtureCase:
    """A deterministic conformance fixture pair."""

    name: str
    method_family: str
    input_artifact: str | None
    expected_output_artifact: str
    tolerance_policy: str
    provenance: dict[str, Any] | None = None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> Any:
    """Load a JSON artifact from disk."""
    return _load_json(path)


def _load_json_payload(path: Path) -> Any:
    """Load a JSON fixture artifact."""
    return _load_json(path)


def _require_pyarrow_backend(suffix: str) -> None:
    """Ensure the optional pyarrow backend is available for binary artifacts."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            f"{suffix} fixture artifacts require the optional 'pyarrow' backend"
        ) from exc


def _load_parquet_payload(path: Path) -> Any:
    """Load a Parquet fixture artifact via the optional Arrow backend."""
    _require_pyarrow_backend(path.suffix.lower())

    import pyarrow.parquet as pq

    return pq.read_table(path)


def _load_arrow_payload(path: Path) -> Any:
    """Load an Arrow IPC fixture artifact via the optional Arrow backend."""
    _require_pyarrow_backend(path.suffix.lower())

    import pyarrow as pa
    from pyarrow import ipc

    with path.open("rb") as handle:
        try:
            return ipc.open_file(handle).read_all()
        except pa.ArrowInvalid:
            handle.seek(0)
            return ipc.open_stream(handle).read_all()


def _load_fixture_payload(path: Path) -> Any:
    """Load a fixture artifact from disk using its suffix."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_payload(path)
    if suffix == ".parquet":
        return _load_parquet_payload(path)
    if suffix == ".arrow":
        return _load_arrow_payload(path)
    raise ValidationError(f"Unsupported fixture artifact format: {path.suffix}")


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
    if schema.get("maximum") is not None and value > schema["maximum"]:
        raise ValidationError(f"{path}: value is above maximum")
    if schema.get("exclusiveMaximum") is not None and value >= schema["exclusiveMaximum"]:
        raise ValidationError(
            f"{path}: value is not below exclusiveMaximum"
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: value is not one of the allowed enum values")


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


def _validate_payload_against_schema(
    payload: Any, schema_path: Path, path: str = "$"
) -> None:
    schema = _load_json(schema_path)
    _validate(payload, schema, path, schema_path)


def _validate_example(example_path: Path, schema_path: Path) -> None:
    example = _load_json(example_path)
    schema = _load_json(schema_path)
    _validate(example, schema, "$", schema_path)


def resolve_fixture_artifact(ref: str) -> Path:
    """Resolve a fixture artifact relative to the v1 fixture root."""
    artifact_path = (FIXTURE_ROOT / ref).resolve()
    try:
        artifact_path.relative_to(FIXTURE_ROOT.resolve())
    except ValueError as exc:
        raise ValidationError(f"artifact escapes fixture root: {ref}") from exc
    return artifact_path


def _resolve_fixture_artifact(ref: str) -> Path:
    return resolve_fixture_artifact(ref)


def load_fixture_manifest(manifest_path: Path = FIXTURE_MANIFEST) -> dict[str, Any]:
    """Load and validate the top-level fixture manifest structure."""
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValidationError("fixture manifest must be an object")
    if manifest.get("version") != "v1":
        raise ValidationError("fixture manifest version must be 'v1'")
    return manifest


def load_fixture_payload(path: Path) -> Any:
    """Load a fixture payload from disk."""
    return _load_fixture_payload(path)


def iter_fixture_cases(
    section: str = "normative",
    manifest_path: Path | None = None,
    validate_payloads: bool = True,
) -> list[FixtureCase]:
    """Return validated fixture cases for a manifest section."""
    if manifest_path is None:
        manifest_path = FIXTURE_MANIFEST
    manifest = load_fixture_manifest(manifest_path)
    value = manifest.get(section)
    if not isinstance(value, list):
        raise ValidationError(f"fixture manifest missing {section} array")

    cases: list[FixtureCase] = []
    seen_names: set[str] = set()
    seen_output_artifacts: set[str] = set()

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValidationError(f"$.{section}[{index}]: expected object")

        entry_path = f"$.{section}[{index}]"
        name = _require_non_empty_string(item.get("name"), f"{entry_path}.name")
        method_family = _require_non_empty_string(
            item.get("method_family"), f"{entry_path}.method_family"
        )
        input_artifact_value = item.get("input_artifact")
        if section == "normative" or input_artifact_value is not None:
            input_artifact = _require_non_empty_string(
                input_artifact_value, f"{entry_path}.input_artifact"
            )
        else:
            input_artifact = None
        output_artifact = _require_non_empty_string(
            item.get("expected_output_artifact"),
            f"{entry_path}.expected_output_artifact",
        )
        tolerance_policy = _require_non_empty_string(
            item.get("tolerance_policy"), f"{entry_path}.tolerance_policy"
        )
        provenance = item.get("provenance")

        if name in seen_names:
            raise ValidationError(f"{entry_path}.name: duplicate entry {name!r}")
        if output_artifact in seen_output_artifacts:
            raise ValidationError(
                f"{entry_path}.expected_output_artifact: duplicate entry {output_artifact!r}"
            )
        seen_names.add(name)
        seen_output_artifacts.add(output_artifact)

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
            raise ValidationError(
                f"{entry_path}.provenance: only normative fixtures may declare provenance"
            )

        if input_artifact is not None:
            input_path = resolve_fixture_artifact(input_artifact)
            if not input_path.is_file():
                raise ValidationError(
                    f"{entry_path}.input_artifact: missing artifact {input_artifact}"
                )
            if validate_payloads:
                load_fixture_payload(input_path)
        output_path = resolve_fixture_artifact(output_artifact)
        if not output_path.is_file():
            raise ValidationError(
                f"{entry_path}.expected_output_artifact: missing artifact {output_artifact}"
            )

        if validate_payloads:
            output_payload = load_fixture_payload(output_path)
            schema_path = RESULT_SCHEMA_BY_METHOD_FAMILY.get(method_family)
            if schema_path is not None:
                _validate_payload_against_schema(output_payload, schema_path)

        cases.append(
            FixtureCase(
                name=name,
                method_family=method_family,
                input_artifact=input_artifact,
                expected_output_artifact=output_artifact,
                tolerance_policy=tolerance_policy,
                provenance=provenance,
            )
        )

    return cases


def validate_fixture_catalog_layout(manifest_path: Path | None = None) -> None:
    """Validate the fixture catalog structure and artifact layout only."""
    iter_fixture_cases("normative", manifest_path, validate_payloads=False)
    iter_fixture_cases("illustrative", manifest_path, validate_payloads=False)


def _validate_fixture_manifest(manifest_path: Path | None = None) -> None:
    iter_fixture_cases("normative", manifest_path)
    iter_fixture_cases("illustrative", manifest_path)


def main() -> int:
    """Validate the committed core API spec examples and fixture catalog."""
    _validate_fixture_manifest()
    print(f"validated {FIXTURE_MANIFEST.relative_to(REPO_ROOT)}")
    for case in iter_fixture_cases():
        if case.input_artifact is not None:
            print(
                f"validated {resolve_fixture_artifact(case.input_artifact).relative_to(REPO_ROOT)}"
            )
        print(
            "validated "
            f"{resolve_fixture_artifact(case.expected_output_artifact).relative_to(REPO_ROOT)}"
        )

    checks = [
        (SCHEMA_ROOT / "decision-problem.schema.json", EXAMPLE_ROOT / "decision-problem.example.json"),
        (SCHEMA_ROOT / "intervention.schema.json", EXAMPLE_ROOT / "intervention.example.json"),
        (SCHEMA_ROOT / "trial-design.schema.json", EXAMPLE_ROOT / "trial-design.example.json"),
        (SCHEMA_ROOT / "parameter-set.schema.json", EXAMPLE_ROOT / "parameter-set.example.json"),
        (SCHEMA_ROOT / "value-array.schema.json", EXAMPLE_ROOT / "value-array.example.json"),
        (SCHEMA_ROOT / "diagnostics.schema.json", EXAMPLE_ROOT / "diagnostics.example.json"),
        (
            SCHEMA_ROOT / "method-metadata.schema.json",
            EXAMPLE_ROOT / "method-metadata.example.json",
        ),
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
