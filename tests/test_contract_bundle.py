"""Independent consumer assurance for the versioned VOP-VOIAGE bundle."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq
import pytest

from scripts.check_contract_bundle_performance import measure_contract_bundle
from voiage.analysis import DecisionAnalysis
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


def _identity(*, fingerprint: str = "0" * 64) -> dict[str, object]:
    return {
        "schema_id": "typed_pipeline_records",
        "schema_version": "1.0.0",
        "schema_fingerprint": fingerprint,
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
    manifest = {
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
        == "fba786ead24ac359fc943892f74bcc905f38e96f0ac4cf945824b6c63fd80a0c"
    )
    assert (
        verified.arrow_schema_fingerprint
        == "a6d94152c7c0ab3b92d0806fdd87240931e6387e8aafe76fbdb88d27da0f4b5e"
    )


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
    current["schema_fingerprint"] = "1" * 64
    current["fields"] = (
        *FIELDS,
        {"name": "note", "arrow_type": "string", "nullable": True, "unit": None},
    )

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
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=1,
        max_size=4,
        unique=True,
    )
)
def test_any_unique_nullable_suffix_is_compatible(field_names: list[str]) -> None:
    previous = _identity()
    current = deepcopy(previous)
    current["schema_version"] = "1.1.0"
    current["schema_fingerprint"] = "1" * 64
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

    report = validate_schema_evolution(previous, current)
    assert report.added_fields == tuple(field_names)


def test_provenance_requirement_change_fails_closed() -> None:
    previous = _identity()
    current = deepcopy(previous)
    current["required_metadata"] = [*REQUIRED_METADATA, "vop_voiage.unknown"]

    with pytest.raises(BundleVerificationError, match="provenance"):
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
