"""Independent consumer assurance for the versioned VOP-VOIAGE bundle."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import TYPE_CHECKING

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq
import pytest

from scripts.check_contract_bundle_performance import measure_contract_bundle
from voiage.analysis import DecisionAnalysis
from voiage.contracts import bundle as bundle_module
from voiage.contracts.bundle import (
    BundleVerificationError,
    ContractPerformanceBudget,
    canonical_bundle_digest,
    validate_schema_evolution,
    verify_contract_bundle,
    verify_pinned_contract_bundle,
)
from voiage.contracts.interchange import schema_fingerprint
from voiage.schema import ValueArray

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

FIELDS = (
    {"name": "analysis_id", "arrow_type": "string", "nullable": False, "unit": None},
    {
        "name": "value",
        "arrow_type": "double",
        "nullable": False,
        "unit": {"symbol_field": None, "dimension": "currency"},
    },
)
RECORDS = [
    {"analysis_id": "a-1", "value": 1.25},
    {"analysis_id": "a-2", "value": 2.5},
]
REQUIRED_METADATA = ["vop_voiage.producer", "vop_voiage.provenance_sha256"]


def _fields_fingerprint(fields: Sequence[Mapping[str, object]]) -> str:
    arrow_types = {"string": pa.string(), "double": pa.float64()}
    return schema_fingerprint(
        pa.schema(
            [
                pa.field(
                    str(field["name"]),
                    arrow_types[str(field["arrow_type"])],
                    nullable=bool(field["nullable"]),
                )
                for field in fields
            ]
        )
    )


def _identity(*, fingerprint: str | None = None) -> dict[str, object]:
    return {
        "schema_id": "typed_pipeline_records",
        "schema_version": "1.0.0",
        "schema_fingerprint": fingerprint or _fields_fingerprint(FIELDS),
        "fields": FIELDS,
        "required_metadata": REQUIRED_METADATA,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def _write_bundle(root: Path) -> None:
    provenance = [{"metadata_status": "known", "source_id": "fixture:test-v1"}]
    provenance_json = _canonical_json(provenance).decode().rstrip("\n")
    metadata = {
        b"vop_voiage.contract_version": b"1.0.0",
        b"vop_voiage.schema_id": b"typed_pipeline_records",
        b"vop_voiage.schema_version": b"1.0.0",
        b"vop_voiage.producer": b"vop_poc_nz",
        b"vop_voiage.method_contract_version": b"1.1.0",
        b"vop_voiage.interchange": b"apache-arrow",
        b"vop_voiage.provenance_json": provenance_json.encode(),
        b"vop_voiage.provenance_sha256": sha256(provenance_json.encode())
        .hexdigest()
        .encode(),
    }
    schema = pa.schema(
        [
            pa.field("analysis_id", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ],
        metadata=metadata,
    )
    fingerprint = schema_fingerprint(schema)
    schema = schema.with_metadata(
        {**metadata, b"vop_voiage.schema_fingerprint": fingerprint.encode()}
    )
    table = pa.Table.from_pylist(RECORDS, schema=schema)

    paths: dict[str, bytes] = {
        "arrow/typed-pipeline-records.schema.json": _canonical_json(
            {
                "schema_id": "typed_pipeline_records",
                "schema_version": "1.0.0",
                "schema_fingerprint": fingerprint,
                "fields": FIELDS,
                "required_metadata": sorted(
                    [key.decode() for key in metadata]
                    + ["vop_voiage.schema_fingerprint"]
                ),
            }
        ),
        "fixtures/typed-pipeline-records.json": _canonical_json(
            {
                "schema_id": "typed_pipeline_records",
                "schema_fingerprint": fingerprint,
                "records": RECORDS,
            }
        ),
        "migration-policy.json": _canonical_json(
            {
                "schema_version": "1.0.0",
                "bundle_id": "vop-voiage-contracts",
                "current_bundle_version": "1.0.0",
                "integrity": "sha256-manifest; unsigned until approved release publication",
                "compatible_changes": ["append_nullable_field"],
                "incompatible_changes": [
                    "change_dtype",
                    "change_nullability",
                    "change_provenance_requirement",
                    "change_schema_id",
                    "change_semantic_unit",
                    "insert_or_reorder_field",
                    "remove_field",
                    "unknown_change",
                ],
                "consumer_rule": "reject every change not explicitly compatible",
            }
        ),
        "references/analytical-reference-manifest.json": _canonical_json(
            {
                "reference_id": "vop-voiage-analytical-reference",
                "reference_version": "1.0.0",
                "replication_status": "producer_authored_pending_external_replication",
            }
        ),
        "schemas/example.schema.json": _canonical_json(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
            }
        ),
    }
    for relative, payload in paths.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    arrow_path = root / "fixtures/typed-pipeline-records.arrow"
    arrow_path.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_file(arrow_path, table.schema) as writer:
        writer.write_table(table)
    parquet_path = root / "fixtures/typed-pipeline-records.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
                "media_type": {
                    ".arrow": "application/vnd.apache.arrow.file",
                    ".json": "application/json",
                    ".parquet": "application/vnd.apache.parquet",
                }[path.suffix],
            }
        )
    reference_entry = next(
        entry
        for entry in entries
        if entry["path"] == "references/analytical-reference-manifest.json"
    )
    manifest = {
        "analytical_reference": {
            "path": reference_entry["path"],
            "reference_id": "vop-voiage-analytical-reference",
            "reference_version": "1.0.0",
            "sha256": reference_entry["sha256"],
        },
        "schema_version": "1.0.0",
        "bundle_id": "vop-voiage-contracts",
        "bundle_version": "1.0.0",
        "contract_version": "1.0.0",
        "method_contract_version": "1.1.0",
        "producer": "vop_poc_nz",
        "source_repository": "edithatogo/vop_poc_nz",
        "source_revision": "contract-bundle/1.0.0",
        "source_path": "contracts/vop-voiage/1.0.0",
        "signature_policy": "unsigned_sha256_manifest",
        "integrity_algorithm": "sha256",
        "schema_source": "schemas/domain",
        "files": entries,
        "bundle_sha256": canonical_bundle_digest(entries),
    }
    (root / "manifest.json").write_bytes(_canonical_json(manifest))


def _rewrite_manifested_json(
    root: Path,
    relative: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    target = root / relative
    document = json.loads(target.read_text(encoding="utf-8"))
    mutate(document)
    target.write_bytes(_canonical_json(document))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    payload = target.read_bytes()
    entry["sha256"] = sha256(payload).hexdigest()
    entry["size"] = len(payload)
    manifest["bundle_sha256"] = canonical_bundle_digest(manifest["files"])
    manifest_path.write_bytes(_canonical_json(manifest))


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "1.0.0"
    _write_bundle(root)
    return root


def test_bundle_verification_is_independent_and_cross_format(bundle: Path) -> None:
    verified = verify_contract_bundle(bundle)

    assert verified.bundle_version == "1.0.0"
    assert verified.producer == "vop_poc_nz"
    assert verified.records == tuple(RECORDS)
    assert verified.arrow_schema_fingerprint == schema_fingerprint(
        verified.arrow_schema
    )
    assert "vop" not in {module.split(".")[0] for module in __import__("sys").modules}


@pytest.mark.parametrize(
    "failure", ["digest", "extra", "traversal", "provenance", "migration"]
)
def test_bundle_tampering_and_unsupported_provenance_fail_closed(
    bundle: Path, failure: str
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if failure == "digest":
        manifest["files"][0]["sha256"] = "0" * 64
    elif failure == "extra":
        (bundle / "unmanifested.txt").write_text("surprise", encoding="utf-8")
    elif failure == "traversal":
        manifest["files"][0]["path"] = "../escape.json"
    elif failure == "provenance":
        manifest["producer"] = "untrusted"
    else:
        policy_path = bundle / "migration-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["consumer_rule"] = "best_effort"
        policy_path.write_bytes(_canonical_json(policy))
        for entry in manifest["files"]:
            if entry["path"] == "migration-policy.json":
                payload = policy_path.read_bytes()
                entry["sha256"] = sha256(payload).hexdigest()
                entry["size"] = len(payload)
        manifest["bundle_sha256"] = canonical_bundle_digest(manifest["files"])
    manifest_path.write_bytes(_canonical_json(manifest))

    with pytest.raises(BundleVerificationError):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-key",
        "bad-semver",
        "manifest-version",
        "integrity",
        "schema-source",
        "source-revision",
        "source-path",
        "empty-files",
        "reference-type",
        "reference-shape",
        "reference-path",
        "reference-id",
        "reference-version",
        "reference-digest",
        "entry-type",
        "entry-shape",
        "backslash-path",
        "duplicate-path",
        "digest-shape",
        "boolean-size",
        "empty-media-type",
        "unsorted-files",
        "reference-entry-digest",
        "aggregate-digest",
    ],
)
def test_manifest_contract_rejects_each_untrusted_shape(
    bundle: Path, failure: str
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["files"][0]
    if failure == "missing-key":
        del manifest["producer"]
    elif failure == "bad-semver":
        manifest["contract_version"] = "v1"
    elif failure == "manifest-version":
        manifest["schema_version"] = "2.0.0"
    elif failure == "integrity":
        manifest["integrity_algorithm"] = "sha1"
    elif failure == "schema-source":
        manifest["schema_source"] = "runtime"
    elif failure == "source-revision":
        manifest["source_revision"] = "main"
    elif failure == "source-path":
        manifest["source_path"] = "contracts/elsewhere"
    elif failure == "empty-files":
        manifest["files"] = []
    elif failure == "reference-type":
        manifest["analytical_reference"] = []
    elif failure == "reference-shape":
        del manifest["analytical_reference"]["reference_id"]
    elif failure == "reference-path":
        manifest["analytical_reference"]["path"] = "references/other.json"
    elif failure == "reference-id":
        manifest["analytical_reference"]["reference_id"] = "other"
    elif failure == "reference-version":
        manifest["analytical_reference"]["reference_version"] = "2.0.0"
    elif failure == "reference-digest":
        manifest["analytical_reference"]["sha256"] = "not-a-digest"
    elif failure == "entry-type":
        manifest["files"][0] = []
    elif failure == "entry-shape":
        del first["media_type"]
    elif failure == "backslash-path":
        first["path"] = "arrow\\schema.json"
    elif failure == "duplicate-path":
        manifest["files"][1]["path"] = first["path"]
    elif failure == "digest-shape":
        first["sha256"] = "invalid"
    elif failure == "boolean-size":
        first["size"] = True
    elif failure == "empty-media-type":
        first["media_type"] = ""
    elif failure == "unsorted-files":
        manifest["files"] = list(reversed(manifest["files"]))
    elif failure == "reference-entry-digest":
        manifest["analytical_reference"]["sha256"] = "0" * 64
    else:
        manifest["bundle_sha256"] = "0" * 64
    manifest_path.write_bytes(_canonical_json(manifest))

    with pytest.raises(BundleVerificationError):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize("failure", ["missing-file", "byte-size", "file-digest"])
def test_manifest_inventory_rejects_missing_or_changed_bytes(
    bundle: Path, failure: str
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][0]
    target = bundle / entry["path"]
    if failure == "missing-file":
        target.unlink()
    elif failure == "byte-size":
        entry["size"] += 1
        manifest["bundle_sha256"] = canonical_bundle_digest(manifest["files"])
        manifest_path.write_bytes(_canonical_json(manifest))
    else:
        target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(BundleVerificationError):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize("payload", [b"{", b"[]"])
def test_manifest_must_be_a_valid_json_object(bundle: Path, payload: bytes) -> None:
    (bundle / "manifest.json").write_bytes(payload)
    with pytest.raises(BundleVerificationError, match="manifest"):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize("path", ["/absolute.json", "dir/../relative.json"])
def test_manifest_paths_reject_noncanonical_posix_forms(
    bundle: Path, path: str
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = path
    manifest_path.write_bytes(_canonical_json(manifest))
    with pytest.raises(BundleVerificationError, match="manifest path"):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize(
    "failure",
    [
        "identity-fingerprint",
        "identity-fields",
        "metadata-name",
        "metadata-duplicate",
        "fixture-schema",
        "fixture-fingerprint",
        "records-type",
        "records-row",
        "records-value",
    ],
)
def test_cross_format_identity_and_record_mismatches_fail_closed(
    bundle: Path, failure: str
) -> None:
    identity_path = "arrow/typed-pipeline-records.schema.json"
    fixture_path = "fixtures/typed-pipeline-records.json"

    def mutate_identity(identity: dict[str, object]) -> None:
        if failure == "identity-fingerprint":
            identity["schema_fingerprint"] = "0" * 64
        elif failure == "identity-fields":
            identity["fields"] = "fields"
        elif failure == "metadata-name":
            identity["required_metadata"] = [1]
        else:
            names = list(identity["required_metadata"])
            identity["required_metadata"] = [*names, names[0]]

    def mutate_fixture(fixture: dict[str, object]) -> None:
        if failure == "fixture-schema":
            fixture["schema_id"] = "other"
        elif failure == "fixture-fingerprint":
            fixture["schema_fingerprint"] = "0" * 64
        elif failure == "records-type":
            fixture["records"] = "records"
        elif failure == "records-row":
            fixture["records"] = [1]
        else:
            records = list(fixture["records"])
            records[0] = {**records[0], "value": 999.0}
            fixture["records"] = records

    if failure.startswith(("identity", "metadata")):
        _rewrite_manifested_json(bundle, identity_path, mutate_identity)
    else:
        _rewrite_manifested_json(bundle, fixture_path, mutate_fixture)
    with pytest.raises(BundleVerificationError):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize(
    "relative",
    [
        "fixtures/typed-pipeline-records.arrow",
        "fixtures/typed-pipeline-records.parquet",
    ],
)
def test_corrupt_columnar_fixture_fails_closed(bundle: Path, relative: str) -> None:
    target = bundle / relative
    target.write_bytes(b"not-arrow")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry["sha256"] = sha256(target.read_bytes()).hexdigest()
    entry["size"] = target.stat().st_size
    manifest["bundle_sha256"] = canonical_bundle_digest(manifest["files"])
    manifest_path.write_bytes(_canonical_json(manifest))

    with pytest.raises(BundleVerificationError, match="fixture"):
        verify_contract_bundle(bundle)


@pytest.mark.parametrize(
    "failure",
    [
        "shape",
        "identity",
        "compatible",
        "incompatible-type",
        "incompatible-set",
    ],
)
def test_migration_policy_rejects_unsupported_semantics(
    bundle: Path, failure: str
) -> None:
    def mutate(policy: dict[str, object]) -> None:
        if failure == "shape":
            policy["unexpected"] = True
        elif failure == "identity":
            policy["current_bundle_version"] = "2.0.0"
        elif failure == "compatible":
            policy["compatible_changes"] = ["remove_field"]
        elif failure == "incompatible-type":
            policy["incompatible_changes"] = "all"
        else:
            policy["incompatible_changes"] = ["change_dtype"]

    _rewrite_manifested_json(bundle, "migration-policy.json", mutate)
    with pytest.raises(BundleVerificationError):
        verify_contract_bundle(bundle)


def test_bundle_can_be_pinned_to_an_expected_digest(bundle: Path) -> None:
    verified = verify_contract_bundle(bundle)
    assert (
        verify_contract_bundle(bundle, expected_bundle_sha256=verified.bundle_sha256)
        == verified
    )
    with pytest.raises(BundleVerificationError, match="pinned"):
        verify_contract_bundle(bundle, expected_bundle_sha256="0" * 64)


def test_checked_in_bundle_matches_immutable_upstream_pin() -> None:
    bundles = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "integration"
        / "vop-voiage"
        / "bundles"
    )
    verified = verify_pinned_contract_bundle(
        bundles / "1.0.0", bundles / "UPSTREAM.json"
    )

    assert (
        verified.bundle_sha256
        == "f79a8d56b22736e34f10d8cb02db46239f27093fd1c366d4ca0ba2c688b60798"
    )
    assert (
        verified.arrow_schema_fingerprint
        == "a6d94152c7c0ab3b92d0806fdd87240931e6387e8aafe76fbdb88d27da0f4b5e"
    )


@pytest.mark.parametrize(
    "failure",
    ["shape", "repository", "commit", "digest", "fingerprint", "version"],
)
def test_upstream_pin_rejects_untrusted_identity(
    bundle: Path, tmp_path: Path, failure: str
) -> None:
    verified = verify_contract_bundle(bundle)
    pin: dict[str, object] = {
        "schema_version": "1.0.0",
        "canonical_repository": "edithatogo/vop_poc_nz",
        "canonical_path": "contracts/vop-voiage/1.0.0",
        "canonical_git_commit": "a" * 40,
        "bundle_version": "1.0.0",
        "bundle_sha256": verified.bundle_sha256,
        "arrow_schema_fingerprint": verified.arrow_schema_fingerprint,
        "signature_policy": "unsigned_sha256_manifest",
    }
    if failure == "shape":
        pin["unexpected"] = True
    elif failure == "repository":
        pin["canonical_repository"] = "attacker/repository"
    elif failure == "commit":
        pin["canonical_git_commit"] = "main"
    elif failure == "digest":
        pin["bundle_sha256"] = "invalid"
    elif failure == "fingerprint":
        pin["arrow_schema_fingerprint"] = "0" * 64
    else:
        pin["bundle_version"] = "1.1.0"
        pin["canonical_path"] = "contracts/vop-voiage/1.1.0"
    pin_path = tmp_path / "UPSTREAM.json"
    pin_path.write_bytes(_canonical_json(pin))

    with pytest.raises(BundleVerificationError):
        verify_pinned_contract_bundle(bundle, pin_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_id": "other"}, "identity"),
        ({"schema_version": "2.0.0"}, "version"),
        ({"fields": ({**FIELDS[0], "arrow_type": "large_string"}, FIELDS[1])}, "type"),
        ({"fields": (FIELDS[0], {**FIELDS[1], "unit": "USD"})}, "unit"),
        ({"fields": (FIELDS[0],)}, "removed"),
    ],
)
def test_incompatible_schema_evolution_fails_closed(
    mutation: dict[str, object], message: str
) -> None:
    previous = _identity()
    current = deepcopy(previous) | mutation

    with pytest.raises(BundleVerificationError, match=message):
        validate_schema_evolution(previous, current)


def test_nullable_addition_is_backward_compatible_and_requires_version_bump() -> None:
    previous = _identity()
    current = deepcopy(previous)
    current["schema_version"] = "1.1.0"
    current["fields"] = (
        *FIELDS,
        {"name": "note", "arrow_type": "string", "nullable": True, "unit": None},
    )
    current["schema_fingerprint"] = _fields_fingerprint(current["fields"])

    report = validate_schema_evolution(previous, current)
    assert report.backward_compatible is True
    assert report.forward_compatible is False
    assert report.added_fields == ("note",)
    current["schema_version"] = "1.0.0"
    with pytest.raises(BundleVerificationError, match="version"):
        validate_schema_evolution(previous, current)


@given(st.permutations(RECORDS))
def test_record_order_is_a_semantics_preserving_metamorphism(
    permutation: list[dict[str, object]],
) -> None:
    assert sorted(permutation, key=lambda row: str(row["analysis_id"])) == sorted(
        RECORDS, key=lambda row: str(row["analysis_id"])
    )


@given(
    st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).filter(
            lambda name: name not in {"analysis_id", "value"}
        ),
        min_size=1,
        max_size=4,
        unique=True,
    )
)
def test_any_unique_nullable_suffix_is_compatible(field_names: list[str]) -> None:
    previous = _identity()
    current = deepcopy(previous)
    current["schema_version"] = "1.1.0"
    current["fields"] = (
        *FIELDS,
        *(
            {
                "name": name,
                "arrow_type": "string",
                "nullable": True,
                "unit": None,
            }
            for name in field_names
        ),
    )
    current["schema_fingerprint"] = _fields_fingerprint(current["fields"])

    report = validate_schema_evolution(previous, current)
    assert report.added_fields == tuple(field_names)


def test_provenance_requirement_change_fails_closed() -> None:
    previous = _identity()
    current = deepcopy(previous)
    current["required_metadata"] = [*REQUIRED_METADATA, "vop_voiage.unknown"]

    with pytest.raises(BundleVerificationError, match="provenance"):
        validate_schema_evolution(previous, current)


@pytest.mark.parametrize(
    "failure",
    [
        "unknown-identity-field",
        "fields-string",
        "field-not-object",
        "field-shape",
        "field-name-type",
        "unit-shape",
        "unit-symbol-type",
        "duplicate-field",
        "metadata-type",
        "metadata-duplicate",
        "reordered",
        "nonnullable-addition",
        "major-version-addition",
        "version-without-change",
        "invalid-fingerprint",
        "unchanged-addition-fingerprint",
        "wrong-previous-fingerprint",
        "unsupported-arrow-type",
    ],
)
def test_schema_evolution_rejects_malformed_and_ambiguous_descriptors(
    failure: str,
) -> None:
    previous = _identity()
    current = deepcopy(previous)
    if failure == "unknown-identity-field":
        current["consumer_hint"] = "ignore"
    elif failure == "fields-string":
        current["fields"] = "not-fields"
    elif failure == "field-not-object":
        current["fields"] = (*FIELDS, "field")
    elif failure == "field-shape":
        current["fields"] = ({**FIELDS[0], "extra": True}, FIELDS[1])
    elif failure == "field-name-type":
        current["fields"] = ({**FIELDS[0], "name": 1}, FIELDS[1])
    elif failure == "unit-shape":
        current["fields"] = (
            FIELDS[0],
            {**FIELDS[1], "unit": {"dimension": "currency"}},
        )
    elif failure == "unit-symbol-type":
        current["fields"] = (
            FIELDS[0],
            {
                **FIELDS[1],
                "unit": {"dimension": "currency", "symbol_field": 1},
            },
        )
    elif failure == "duplicate-field":
        current["fields"] = (FIELDS[0], FIELDS[0])
    elif failure == "metadata-type":
        current["required_metadata"] = "metadata"
    elif failure == "metadata-duplicate":
        current["required_metadata"] = [*REQUIRED_METADATA, REQUIRED_METADATA[0]]
    elif failure == "reordered":
        current["fields"] = tuple(reversed(FIELDS))
    elif failure == "nonnullable-addition":
        current["schema_version"] = "1.1.0"
        current["fields"] = (
            *FIELDS,
            {"name": "note", "arrow_type": "string", "nullable": False, "unit": None},
        )
    elif failure == "major-version-addition":
        current["schema_version"] = "2.0.0"
        current["fields"] = (
            *FIELDS,
            {"name": "note", "arrow_type": "string", "nullable": True, "unit": None},
        )
        current["schema_fingerprint"] = _fields_fingerprint(current["fields"])
    elif failure == "version-without-change":
        current["schema_version"] = "1.1.0"
    elif failure == "invalid-fingerprint":
        current["schema_fingerprint"] = "invalid"
    elif failure == "unchanged-addition-fingerprint":
        current["schema_version"] = "1.1.0"
        current["fields"] = (
            *FIELDS,
            {"name": "note", "arrow_type": "string", "nullable": True, "unit": None},
        )
    elif failure == "wrong-previous-fingerprint":
        previous["schema_fingerprint"] = "0" * 64
    else:
        current["schema_version"] = "1.1.0"
        current["fields"] = (
            *FIELDS,
            {"name": "note", "arrow_type": "binary", "nullable": True, "unit": None},
        )
        current["schema_fingerprint"] = "1" * 64

    with pytest.raises(BundleVerificationError):
        validate_schema_evolution(previous, current)


def test_checked_in_previous_current_migration_is_anchored_to_pinned_identity() -> None:
    bundles = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "integration"
        / "vop-voiage"
        / "bundles"
    )
    migration = json.loads(
        (bundles / "migrations" / "1.0.0-to-1.1.0.json").read_text(encoding="utf-8")
    )
    pinned = json.loads(
        (bundles / "1.0.0" / "arrow" / "typed-pipeline-records.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert migration["previous"] == pinned
    report = validate_schema_evolution(migration["previous"], migration["current"])
    assert report.backward_compatible is True
    assert report.forward_compatible is False
    assert report.added_fields == ("decision_context",)


@given(
    st.lists(
        st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=6,
        max_size=6,
    )
)
@settings(max_examples=5, deadline=None)
def test_numpy_and_jax_evpi_are_differentially_conformant(values: list[float]) -> None:
    value_array = ValueArray.from_numpy(
        np.asarray(values, dtype=np.float32).reshape(3, 2), ["A", "B"]
    )

    numpy_value = DecisionAnalysis(nb_array=value_array, backend="numpy").evpi()
    jax_value = DecisionAnalysis(nb_array=value_array, backend="jax").evpi()

    float32_scale = max(1.0, float(np.max(np.abs(value_array.values))))
    float32_atol = float(np.finfo(np.float32).eps) * float32_scale
    assert numpy_value == pytest.approx(jax_value, rel=1e-6, abs=float32_atol)


def test_bundle_performance_budget_contract_accepts_and_rejects_metrics() -> None:
    budget = ContractPerformanceBudget(
        cpu_seconds=0.25,
        peak_memory_bytes=32 * 1024 * 1024,
        allocation_count=50_000,
        serialization_bytes=128 * 1024,
    )
    budget.enforce(
        cpu_seconds=0.1,
        peak_memory_bytes=1024,
        allocation_count=100,
        serialization_bytes=2048,
    )

    with pytest.raises(BundleVerificationError, match="cpu_seconds"):
        budget.enforce(
            cpu_seconds=0.3,
            peak_memory_bytes=1024,
            allocation_count=100,
            serialization_bytes=2048,
        )


def test_bundle_performance_measurement_retains_failure_evidence(tmp_path) -> None:
    bundles = (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "integration"
        / "vop-voiage"
        / "bundles"
    )
    budget_path = tmp_path / "impossible-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "verification_repetitions": 1,
                "cpu_seconds": 1e-12,
                "peak_memory_bytes": 1,
                "allocation_count": 1,
                "serialization_bytes": 1,
            }
        ),
        encoding="utf-8",
    )

    evidence = measure_contract_bundle(
        bundle=bundles / "1.0.0",
        pin=bundles / "UPSTREAM.json",
        budget_path=budget_path,
    )

    assert evidence["status"] == "fail"
    assert evidence["measurements"]["serialization_bytes"] > 0
    assert "violation" in evidence


def test_fixture_bundle_stays_within_measured_budget(bundle: Path) -> None:
    budget = ContractPerformanceBudget(
        cpu_seconds=2.0,
        peak_memory_bytes=64 * 1024 * 1024,
        allocation_count=100_000,
        serialization_bytes=256 * 1024,
    )
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    start = perf_counter()
    verified = verify_contract_bundle(bundle)
    elapsed = perf_counter() - start
    after = tracemalloc.take_snapshot()
    _, peak = tracemalloc.get_traced_memory()
    allocations = sum(stat.count_diff for stat in after.compare_to(before, "lineno"))
    tracemalloc.stop()

    budget.enforce(
        cpu_seconds=elapsed,
        peak_memory_bytes=peak,
        allocation_count=allocations,
        serialization_bytes=sum(
            path.stat().st_size for path in bundle.rglob("*") if path.is_file()
        ),
    )
    assert len(verified.records) == 2


def test_low_level_bundle_guards_cover_rejected_runtime_shapes(tmp_path: Path) -> None:
    assert bundle_module._valid_provenance_record([]) is False
    assert bundle_module._valid_provenance_record({"metadata_status": "known"}) is False

    not_directory = tmp_path / "file"
    not_directory.write_text("x", encoding="utf-8")
    with pytest.raises(BundleVerificationError, match="must be a directory"):
        verify_contract_bundle(not_directory)

    with pytest.raises(ValueError, match="positive"):
        ContractPerformanceBudget(0, 1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        ContractPerformanceBudget(1, 0, 1, 1)


def test_arrow_logical_field_guards_are_fail_closed() -> None:
    schema = pa.schema([pa.field("value", pa.float64(), nullable=False)])
    with pytest.raises(BundleVerificationError, match="fields do not match"):
        bundle_module._logical_fields(schema, [])
    with pytest.raises(BundleVerificationError, match="must be an object"):
        bundle_module._logical_fields(schema, ["value"])
    with pytest.raises(BundleVerificationError, match="identity mismatch"):
        bundle_module._logical_fields(
            schema,
            [
                {
                    "name": "other",
                    "arrow_type": "double",
                    "nullable": False,
                    "unit": None,
                }
            ],
        )


def test_runtime_protocol_default_methods_are_behaviorally_exercised() -> None:
    class SchemaDefault(bundle_module._ArrowSchema):
        metadata = None

        def __iter__(self):
            return iter(())

    class ReaderContextDefault(bundle_module._ArrowReaderContext):
        def __enter__(self):
            return None

    assert SchemaDefault().__len__() is None
    assert ReaderContextDefault().__exit__(None, None, None) is None


def test_manifest_bundle_version_runtime_guard_is_fail_closed(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = bundle_module._semver

    def tolerate_only_bundle_version(value: object, label: str) -> tuple[int, int, int]:
        if label == "bundle_version":
            return (1, 0, 0)
        return original(value, label)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_version"] = 1
    manifest_path.write_bytes(_canonical_json(manifest))
    monkeypatch.setattr(bundle_module, "_semver", tolerate_only_bundle_version)
    with pytest.raises(BundleVerificationError, match="must be a string"):
        verify_contract_bundle(bundle)


def test_inventory_guards_reject_symlinks_and_same_size_digest_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inventory"
    root.mkdir()
    target = root / "value.bin"
    target.write_bytes(b"good")
    entry = {
        "path": "value.bin",
        "size": 4,
        "sha256": sha256(b"good").hexdigest(),
        "media_type": "application/octet-stream",
    }
    target.write_bytes(b"evil")
    with pytest.raises(BundleVerificationError, match="SHA-256 mismatch"):
        bundle_module._verify_inventory(root, [entry])

    target.unlink()
    external = tmp_path / "external.bin"
    external.write_bytes(b"good")
    try:
        target.symlink_to(external)
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows host")
    with pytest.raises(BundleVerificationError, match="unsafe bundled file"):
        bundle_module._verify_inventory(root, [entry])


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("parquet-schema", "schema identities differ"),
        ("required-metadata-type", "required_metadata must be an array"),
        ("missing-metadata", "required metadata mismatch"),
        ("producer", "producer provenance mismatch"),
        ("embedded-fingerprint", "embedded fingerprint mismatch"),
        ("provenance-digest", "provenance digest mismatch"),
        ("empty-provenance", "unsupported or non-canonical"),
        ("invalid-provenance", "unsupported or non-canonical"),
        ("noncanonical-provenance", "unsupported or non-canonical"),
        ("record-divergence", "fixture records differ"),
    ],
)
def test_verified_bundle_rejects_each_columnar_identity_divergence(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    arrow_path = bundle / "fixtures" / "typed-pipeline-records.arrow"
    with ipc.open_file(arrow_path) as reader:
        arrow_table = reader.read_all()
    parquet_table = pq.read_table(
        bundle / "fixtures" / "typed-pipeline-records.parquet"
    )

    if failure == "required-metadata-type":
        _rewrite_manifested_json(
            bundle,
            "arrow/typed-pipeline-records.schema.json",
            lambda identity: identity.update(required_metadata="invalid"),
        )
    elif failure == "parquet-schema":
        parquet_table = parquet_table.append_column("extra", pa.array([1, 2]))
    elif failure == "record-divergence":
        parquet_table = pa.Table.from_pylist(
            [{"analysis_id": "other", "value": 1.25}, RECORDS[1]],
            schema=parquet_table.schema,
        )
    else:
        metadata = dict(arrow_table.schema.metadata or {})
        if failure == "missing-metadata":
            metadata.pop(b"vop_voiage.producer")
        elif failure == "producer":
            metadata[b"vop_voiage.producer"] = b"other"
        elif failure == "embedded-fingerprint":
            metadata[b"vop_voiage.schema_fingerprint"] = b"0" * 64
        elif failure == "provenance-digest":
            metadata[b"vop_voiage.provenance_sha256"] = b"0" * 64
        else:
            provenance = {
                "empty-provenance": "[]",
                "invalid-provenance": '[{"metadata_status":"unknown","source_id":"x"}]',
                "noncanonical-provenance": '[ {"metadata_status":"known","source_id":"x"} ]',
            }[failure]
            metadata[b"vop_voiage.provenance_json"] = provenance.encode()
            metadata[b"vop_voiage.provenance_sha256"] = (
                sha256(provenance.encode()).hexdigest().encode()
            )
        arrow_table = arrow_table.replace_schema_metadata(metadata)

    monkeypatch.setattr(bundle_module, "_load_arrow_table", lambda _path: arrow_table)
    monkeypatch.setattr(
        bundle_module, "_read_parquet_table", lambda _path: parquet_table
    )
    with pytest.raises(BundleVerificationError, match=message):
        verify_contract_bundle(bundle)
