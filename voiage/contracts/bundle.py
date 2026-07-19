"""Fail-closed consumption of independently published VOP-VOIAGE bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Self

import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq

from voiage.contracts.interchange import schema_fingerprint

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_MANIFEST_KEYS = {
    "schema_version",
    "bundle_id",
    "bundle_version",
    "contract_version",
    "method_contract_version",
    "producer",
    "source_repository",
    "source_revision",
    "source_path",
    "signature_policy",
    "integrity_algorithm",
    "schema_source",
    "files",
    "bundle_sha256",
}
_ENTRY_KEYS = {"path", "sha256", "size", "media_type"}
_EXPECTED_PROVENANCE = {
    "bundle_id": "vop-voiage-contracts",
    "producer": "vop_poc_nz",
    "source_repository": "edithatogo/vop_poc_nz",
    "signature_policy": "unsigned_sha256_manifest",
}
_MIGRATION_KEYS = {
    "schema_version",
    "bundle_id",
    "current_bundle_version",
    "integrity",
    "compatible_changes",
    "incompatible_changes",
    "consumer_rule",
}
_INCOMPATIBLE_CHANGES = {
    "change_dtype",
    "change_nullability",
    "change_provenance_requirement",
    "change_schema_id",
    "change_semantic_unit",
    "insert_or_reorder_field",
    "remove_field",
    "unknown_change",
}
_PIN_KEYS = {
    "schema_version",
    "canonical_repository",
    "canonical_path",
    "canonical_git_commit",
    "bundle_version",
    "bundle_sha256",
    "arrow_schema_fingerprint",
    "signature_policy",
}


class BundleVerificationError(ValueError):
    """Raised when bundle bytes or compatibility identity are not trustworthy."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def canonical_bundle_digest(entries: Sequence[Mapping[str, object]]) -> str:
    """Hash manifest file entries after deterministic POSIX-path ordering."""
    ordered = sorted(
        (dict(entry) for entry in entries), key=lambda item: str(item.get("path", ""))
    )
    return sha256(_canonical_json(ordered) + b"\n").hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleVerificationError(f"{label} must be a JSON object")
    return value


def _semver(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or _SEMVER.fullmatch(value) is None:
        raise BundleVerificationError(f"{label} must be semantic version x.y.z")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BundleVerificationError("manifest path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleVerificationError(f"unsafe manifest path: {value}")
    return path


def _validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError(
            "manifest fields do not match bundle schema 1.0.0"
        )
    for key in (
        "schema_version",
        "bundle_version",
        "contract_version",
        "method_contract_version",
    ):
        _semver(manifest[key], key)
    if manifest["schema_version"] != "1.0.0":
        raise BundleVerificationError("unsupported manifest schema version")
    if manifest["integrity_algorithm"] != "sha256":
        raise BundleVerificationError("unsupported integrity algorithm")
    if manifest["schema_source"] != "schemas/domain":
        raise BundleVerificationError("unsupported schema source")
    for key, expected in _EXPECTED_PROVENANCE.items():
        if manifest[key] != expected:
            raise BundleVerificationError(f"unsupported provenance field {key}")
    version = manifest["bundle_version"]
    if manifest["source_revision"] != f"contract-bundle/{version}":
        raise BundleVerificationError("unsupported provenance source_revision")
    if manifest["source_path"] != f"contracts/vop-voiage/{version}":
        raise BundleVerificationError("unsupported provenance source_path")
    raw_entries = manifest["files"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise BundleVerificationError("manifest files must be a non-empty array")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != _ENTRY_KEYS:
            raise BundleVerificationError("manifest file entry has unsupported fields")
        relative = _safe_relative_path(raw_entry["path"]).as_posix()
        digest = raw_entry["sha256"]
        size = raw_entry["size"]
        media_type = raw_entry["media_type"]
        if relative in seen:
            raise BundleVerificationError(f"duplicate manifest path: {relative}")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise BundleVerificationError(f"invalid SHA-256 for {relative}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleVerificationError(f"invalid byte size for {relative}")
        if not isinstance(media_type, str) or not media_type:
            raise BundleVerificationError(f"invalid media type for {relative}")
        seen.add(relative)
        entries.append(dict(raw_entry))
    if [entry["path"] for entry in entries] != sorted(seen):
        raise BundleVerificationError("manifest file entries must be path-sorted")
    actual_digest = canonical_bundle_digest(entries)
    if manifest["bundle_sha256"] != actual_digest:
        raise BundleVerificationError("bundle SHA-256 does not match manifest entries")
    return entries


def _verify_inventory(root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    expected = {str(entry["path"]) for entry in entries}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != expected:
        raise BundleVerificationError(
            "bundle inventory does not exactly match manifest"
        )
    for entry in entries:
        relative = str(entry["path"])
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise BundleVerificationError(f"unsafe bundled file {relative}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise BundleVerificationError(
                f"cannot read bundled file {relative}"
            ) from exc
        if len(payload) != entry["size"]:
            raise BundleVerificationError(f"byte size mismatch for {relative}")
        if sha256(payload).hexdigest() != entry["sha256"]:
            raise BundleVerificationError(f"SHA-256 mismatch for {relative}")


def _validate_migration_policy(
    policy: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if set(policy) != _MIGRATION_KEYS:
        raise BundleVerificationError("migration policy has unsupported fields")
    expected_scalars = {
        "schema_version": "1.0.0",
        "bundle_id": manifest["bundle_id"],
        "current_bundle_version": manifest["bundle_version"],
        "integrity": "sha256-manifest; unsigned until approved release publication",
        "consumer_rule": "reject every change not explicitly compatible",
    }
    if any(policy.get(key) != value for key, value in expected_scalars.items()):
        raise BundleVerificationError("migration policy identity is incompatible")
    if policy.get("compatible_changes") != ["append_nullable_field"]:
        raise BundleVerificationError(
            "migration policy compatible changes are unsupported"
        )
    incompatible = policy.get("incompatible_changes")
    if not isinstance(incompatible, list) or set(incompatible) != _INCOMPATIBLE_CHANGES:
        raise BundleVerificationError(
            "migration policy must enumerate every fail-closed change"
        )


def _load_arrow_table(path: Path) -> pa.Table:
    try:
        with ipc.open_file(path) as reader:
            return reader.read_all()
    except (OSError, pa.ArrowException) as exc:
        raise BundleVerificationError(f"invalid Arrow fixture: {exc}") from exc


def _logical_fields(schema: pa.Schema, descriptor: Sequence[object]) -> None:
    if len(schema) != len(descriptor):
        raise BundleVerificationError("Arrow fields do not match identity descriptor")
    for field, expected in zip(schema, descriptor, strict=True):
        if not isinstance(expected, dict):
            raise BundleVerificationError("Arrow field identity must be an object")
        logical = {
            "name": field.name,
            "arrow_type": str(field.type),
            "nullable": field.nullable,
            "unit": expected.get("unit"),
        }
        if set(expected) != set(logical) or expected != logical:
            raise BundleVerificationError(
                f"Arrow field identity mismatch for {field.name}"
            )


@dataclass(frozen=True, slots=True)
class VerifiedContractBundle:
    """Verified in-memory view of a VOP-produced contract bundle."""

    root: Path
    bundle_version: str
    producer: str
    bundle_sha256: str
    arrow_schema: pa.Schema
    arrow_schema_fingerprint: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SchemaEvolutionReport:
    """Compatibility result for one verified Arrow identity transition."""

    backward_compatible: bool
    forward_compatible: bool
    added_fields: tuple[str, ...]


def verify_contract_bundle(
    root: Path, *, expected_bundle_sha256: str | None = None
) -> VerifiedContractBundle:
    """Verify a bundle's provenance, exact bytes, Arrow identity and fixtures."""
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise BundleVerificationError("bundle root must be a directory")
    manifest = _read_json_object(resolved / "manifest.json", "manifest")
    entries = _validate_manifest(manifest)
    _verify_inventory(resolved, entries)
    if (
        expected_bundle_sha256 is not None
        and manifest["bundle_sha256"] != expected_bundle_sha256
    ):
        raise BundleVerificationError("bundle SHA-256 does not match pinned digest")

    migration_policy = _read_json_object(
        resolved / "migration-policy.json", "migration policy"
    )
    _validate_migration_policy(migration_policy, manifest)

    identity = _read_json_object(
        resolved / "arrow" / "typed-pipeline-records.schema.json", "Arrow identity"
    )
    fixture = _read_json_object(
        resolved / "fixtures" / "typed-pipeline-records.json", "JSON fixture"
    )
    arrow_table = _load_arrow_table(
        resolved / "fixtures" / "typed-pipeline-records.arrow"
    )
    try:
        parquet_table = pq.read_table(
            resolved / "fixtures" / "typed-pipeline-records.parquet"
        )
    except (OSError, pa.ArrowException) as exc:
        raise BundleVerificationError(f"invalid Parquet fixture: {exc}") from exc

    fingerprint = schema_fingerprint(arrow_table.schema)
    if schema_fingerprint(parquet_table.schema) != fingerprint:
        raise BundleVerificationError("Arrow and Parquet schema identities differ")
    declared_fingerprint = identity.get("schema_fingerprint")
    if declared_fingerprint != fingerprint:
        raise BundleVerificationError("Arrow schema fingerprint mismatch")
    if fixture.get("schema_id") != identity.get("schema_id"):
        raise BundleVerificationError("JSON fixture schema identity mismatch")
    if fixture.get("schema_fingerprint") != fingerprint:
        raise BundleVerificationError("JSON fixture schema fingerprint mismatch")
    descriptor = identity.get("fields")
    if not isinstance(descriptor, list):
        raise BundleVerificationError("Arrow identity fields must be an array")
    _logical_fields(arrow_table.schema, descriptor)

    required_metadata = identity.get("required_metadata")
    if not isinstance(required_metadata, list) or not all(
        isinstance(key, str) for key in required_metadata
    ):
        raise BundleVerificationError(
            "Arrow required_metadata must be an array of names"
        )
    if required_metadata != sorted(set(required_metadata)):
        raise BundleVerificationError(
            "Arrow required_metadata must be sorted and unique"
        )
    actual_metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (arrow_table.schema.metadata or {}).items()
    }
    if any(key not in actual_metadata for key in required_metadata):
        raise BundleVerificationError("Arrow required metadata mismatch")
    if actual_metadata.get("vop_voiage.producer") != manifest["producer"]:
        raise BundleVerificationError("Arrow producer provenance mismatch")
    if actual_metadata.get("vop_voiage.schema_fingerprint") != fingerprint:
        raise BundleVerificationError("Arrow embedded fingerprint mismatch")
    provenance_json = actual_metadata.get("vop_voiage.provenance_json", "")
    if (
        actual_metadata.get("vop_voiage.provenance_sha256")
        != sha256(provenance_json.encode()).hexdigest()
    ):
        raise BundleVerificationError("Arrow provenance digest mismatch")
    try:
        arrow_provenance = json.loads(provenance_json)
    except json.JSONDecodeError as exc:
        raise BundleVerificationError("Arrow provenance is not canonical JSON") from exc
    if (
        not isinstance(arrow_provenance, list)
        or not arrow_provenance
        or any(
            not isinstance(record, dict)
            or set(record) != {"metadata_status", "source_id"}
            or record["metadata_status"] != "known"
            or not isinstance(record["source_id"], str)
            or not record["source_id"]
            for record in arrow_provenance
        )
        or _canonical_json(arrow_provenance).decode() != provenance_json
    ):
        raise BundleVerificationError(
            "Arrow provenance is unsupported or non-canonical"
        )
    if arrow_table.to_pylist() != parquet_table.to_pylist():
        raise BundleVerificationError("Arrow and Parquet fixture records differ")
    records = fixture.get("records")
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise BundleVerificationError(
            "JSON fixture records must be an array of objects"
        )
    if records != arrow_table.to_pylist():
        raise BundleVerificationError("JSON, Arrow and Parquet fixture records differ")

    return VerifiedContractBundle(
        root=resolved,
        bundle_version=manifest["bundle_version"],
        producer=manifest["producer"],
        bundle_sha256=manifest["bundle_sha256"],
        arrow_schema=arrow_table.schema,
        arrow_schema_fingerprint=fingerprint,
        records=tuple(dict(row) for row in records),
    )


def verify_pinned_contract_bundle(root: Path, pin_path: Path) -> VerifiedContractBundle:
    """Verify a local bundle against its immutable upstream provenance pin."""
    pin = _read_json_object(pin_path, "upstream bundle pin")
    if set(pin) != _PIN_KEYS:
        raise BundleVerificationError("upstream bundle pin has unsupported fields")
    version = pin.get("bundle_version")
    expected = {
        "schema_version": "1.0.0",
        "canonical_repository": "edithatogo/vop_poc_nz",
        "canonical_path": f"contracts/vop-voiage/{version}",
        "signature_policy": "unsigned_sha256_manifest",
    }
    if any(pin.get(key) != value for key, value in expected.items()):
        raise BundleVerificationError("upstream bundle pin provenance is unsupported")
    commit = pin.get("canonical_git_commit")
    digest = pin.get("bundle_sha256")
    fingerprint = pin.get("arrow_schema_fingerprint")
    if (
        not isinstance(commit, str)
        or _GIT_SHA.fullmatch(commit) is None
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or not isinstance(fingerprint, str)
        or _SHA256.fullmatch(fingerprint) is None
    ):
        raise BundleVerificationError("upstream bundle pin digest is invalid")
    verified = verify_contract_bundle(root, expected_bundle_sha256=digest)
    if verified.bundle_version != version:
        raise BundleVerificationError("pinned bundle version mismatch")
    if verified.arrow_schema_fingerprint != fingerprint:
        raise BundleVerificationError("pinned Arrow schema fingerprint mismatch")
    return verified


def _field_map(
    value: Mapping[str, Any],
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    fields = value.get("fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise BundleVerificationError("schema fields must be an array")
    names: list[str] = []
    mapped: dict[str, Mapping[str, Any]] = {}
    for field in fields:
        if (
            not isinstance(field, Mapping)
            or set(field) != {"name", "arrow_type", "nullable", "unit"}
            or not isinstance(field.get("name"), str)
            or not isinstance(field.get("arrow_type"), str)
            or not isinstance(field.get("nullable"), bool)
        ):
            raise BundleVerificationError("schema field identity is invalid")
        unit = field.get("unit")
        if unit is not None and (
            not isinstance(unit, Mapping)
            or set(unit) != {"symbol_field", "dimension"}
            or (
                unit.get("symbol_field") is not None
                and not isinstance(unit.get("symbol_field"), str)
            )
            or not isinstance(unit.get("dimension"), str)
        ):
            raise BundleVerificationError("schema field unit identity is invalid")
        name = str(field["name"])
        if name in mapped:
            raise BundleVerificationError(f"duplicate field identity: {name}")
        names.append(name)
        mapped[name] = field
    return names, mapped


def validate_schema_evolution(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> SchemaEvolutionReport:
    """Accept additive nullable evolution and reject semantic identity drift."""
    identity_keys = {
        "schema_id",
        "schema_version",
        "schema_fingerprint",
        "fields",
        "required_metadata",
    }
    if set(previous) != identity_keys or set(current) != identity_keys:
        raise BundleVerificationError("schema identity has unsupported fields")
    if previous.get("schema_id") != current.get("schema_id"):
        raise BundleVerificationError("schema identity changed")
    previous_version = _semver(
        previous.get("schema_version"), "previous schema version"
    )
    current_version = _semver(current.get("schema_version"), "current schema version")
    previous_names, previous_fields = _field_map(previous)
    current_names, current_fields = _field_map(current)
    previous_metadata = previous.get("required_metadata")
    current_metadata = current.get("required_metadata")
    if (
        not isinstance(previous_metadata, list)
        or not all(isinstance(key, str) for key in previous_metadata)
        or previous_metadata != sorted(set(previous_metadata))
        or current_metadata != previous_metadata
    ):
        raise BundleVerificationError("schema provenance requirements changed")
    if current_names[: len(previous_names)] != previous_names:
        missing = set(previous_names) - set(current_names)
        detail = "removed" if missing else "reordered"
        raise BundleVerificationError(f"schema fields were {detail}")
    for name in previous_names:
        before = previous_fields[name]
        after = current_fields[name]
        for property_name in ("arrow_type", "nullable", "unit"):
            if before.get(property_name) != after.get(property_name):
                raise BundleVerificationError(
                    f"field {property_name} changed for {name}"
                )
    additions = current_names[len(previous_names) :]
    if any(current_fields[name].get("nullable") is not True for name in additions):
        raise BundleVerificationError("added fields must be nullable")
    if additions:
        if (
            current_version <= previous_version
            or current_version[0] != previous_version[0]
        ):
            raise BundleVerificationError(
                "compatible schema change requires a version bump"
            )
    elif current_version != previous_version:
        raise BundleVerificationError("schema version changed without a schema change")
    previous_fingerprint = previous.get("schema_fingerprint")
    current_fingerprint = current.get("schema_fingerprint")
    if (
        not isinstance(previous_fingerprint, str)
        or _SHA256.fullmatch(previous_fingerprint) is None
        or not isinstance(current_fingerprint, str)
        or _SHA256.fullmatch(current_fingerprint) is None
        or (bool(additions) == (previous_fingerprint == current_fingerprint))
    ):
        raise BundleVerificationError("schema fingerprint transition is invalid")
    return SchemaEvolutionReport(
        backward_compatible=True,
        forward_compatible=not additions,
        added_fields=tuple(additions),
    )


@dataclass(frozen=True, slots=True)
class ContractPerformanceBudget:
    """Portable upper bounds for the small deterministic bundle verification lane."""

    cpu_seconds: float
    peak_memory_bytes: int
    allocation_count: int
    serialization_bytes: int

    def __post_init__(self) -> None:
        """Reject nonsensical zero or negative budget limits."""
        if (
            self.cpu_seconds <= 0
            or min(
                self.peak_memory_bytes, self.allocation_count, self.serialization_bytes
            )
            <= 0
        ):
            raise ValueError("performance budgets must be positive")

    def enforce(
        self,
        *,
        cpu_seconds: float,
        peak_memory_bytes: int,
        allocation_count: int,
        serialization_bytes: int,
    ) -> Self:
        """Raise with the precise regressed metric, otherwise return this budget."""
        measurements = {
            "cpu_seconds": cpu_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "allocation_count": allocation_count,
            "serialization_bytes": serialization_bytes,
        }
        for metric, measured in measurements.items():
            if measured > getattr(self, metric):
                raise BundleVerificationError(
                    f"performance budget exceeded: {metric}={measured!r} > {getattr(self, metric)!r}"
                )
        return self
