"""Arrow interchange for canonical v2 analysis-result envelopes."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import json
from typing import TYPE_CHECKING, TypeVar, cast

import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq

from voiage import _runtime
from voiage.contracts.analysis import AnalysisResult, ContractModel

if TYPE_CHECKING:
    from pathlib import Path

PayloadT = TypeVar("PayloadT", bound=ContractModel)
Normalizer = Callable[[object], dict[str, object]]


def schema_fingerprint(schema: pa.Schema) -> str:
    """Return the canonical SHA-256 identity of an Arrow logical schema."""
    fields = [
        {"arrow_type": str(field.type), "name": field.name, "nullable": field.nullable}
        for field in schema.remove_metadata()
    ]
    canonical = json.dumps(fields, separators=(",", ":"), sort_keys=True)
    return sha256(canonical.encode()).hexdigest()


def _with_interchange_marker(table: pa.Table, marker: bytes) -> pa.Table:
    """Replace only the shared container marker while preserving identity."""
    return table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"voiage.interchange": marker,
            b"vop_voiage.interchange": marker,
        }
    )


def _canonical_payload_json(payload: Mapping[str, object]) -> str:
    """Return deterministic transport JSON after Rust validation."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _contract_table(
    payload: object,
    *,
    contract: str,
    schema: pa.Schema,
    normalize: Normalizer,
    columns: Callable[[dict[str, object]], dict[str, list[object]]],
) -> pa.Table:
    """Build a one-row, Rust-validated canonical contract table."""
    normalized = normalize(payload)
    values = columns(normalized)
    values["payload_json"] = [_canonical_payload_json(normalized)]
    table = pa.table(values, schema=schema)
    fingerprint = schema_fingerprint(table.schema)
    return table.replace_schema_metadata(
        {
            b"voiage.contract": contract.encode("ascii"),
            b"voiage.schema_version": b"1.0.0",
            b"voiage.arrow_schema_version": b"1.0.0",
            b"voiage.schema_fingerprint": fingerprint.encode("ascii"),
            b"voiage.producer": b"voiage-rust-validated",
            b"voiage.interchange": b"apache-arrow",
        }
    )


_DECISION_PROBLEM_SCHEMA = pa.schema(
    [
        pa.field("decision_problem_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("analysis_type", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("willingness_to_pay", pa.float64(), nullable=False),
        pa.field("outcome_names", pa.list_(pa.string()), nullable=True),
        pa.field("intervention_count", pa.uint64(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
    ]
)


def decision_problem_table(payload: object) -> pa.Table:
    """Return a Rust-validated canonical v1 Decision Problem Arrow table."""

    def columns(normalized: dict[str, object]) -> dict[str, list[object]]:
        interventions = cast("list[object]", normalized["interventions"])
        return {
            "decision_problem_id": [normalized["decision_problem_id"]],
            "title": [normalized["title"]],
            "analysis_type": [normalized["analysis_type"]],
            "currency": [normalized["currency"]],
            "willingness_to_pay": [normalized["willingness_to_pay"]],
            "outcome_names": [normalized.get("outcome_names")],
            "intervention_count": [len(interventions)],
        }

    return _contract_table(
        payload,
        contract="decision-problem",
        schema=_DECISION_PROBLEM_SCHEMA,
        normalize=_runtime.normalize_decision_problem_json,
        columns=columns,
    )


_STATISTICAL_ASSURANCE_SCHEMA = pa.schema(
    [
        pa.field("reporting_class", pa.string(), nullable=False),
        pa.field("replications", pa.uint64(), nullable=False),
        pa.field("stopping_reason", pa.string(), nullable=False),
        pa.field("has_confidence_interval", pa.bool_(), nullable=False),
        pa.field("has_convergence_evidence", pa.bool_(), nullable=False),
        pa.field("has_rng_identity", pa.bool_(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
    ]
)


def statistical_assurance_table(payload: object) -> pa.Table:
    """Return a Rust-validated canonical v1 assurance Arrow table."""

    def columns(normalized: dict[str, object]) -> dict[str, list[object]]:
        return {
            "reporting_class": [normalized["reporting_class"]],
            "replications": [normalized["replications"]],
            "stopping_reason": [normalized["stopping_reason"]],
            "has_confidence_interval": [normalized["confidence_interval"] is not None],
            "has_convergence_evidence": [normalized["convergence"] is not None],
            "has_rng_identity": [normalized["rng"] is not None],
        }

    return _contract_table(
        payload,
        contract="statistical-assurance",
        schema=_STATISTICAL_ASSURANCE_SCHEMA,
        normalize=_runtime.normalize_statistical_assurance_json,
        columns=columns,
    )


def analysis_result_table[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT],
) -> pa.Table:
    """Return a one-row Arrow table with stable contract metadata."""
    result_json = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    table = pa.table(
        {
            "analysis_id": [result.analysis_id],
            "method_family": [result.method_family],
            "schema_version": [result.schema_version],
            "result_json": [result_json],
        }
    )
    fingerprint = schema_fingerprint(table.schema)
    return table.replace_schema_metadata(
        {
            b"voiage.contract": b"analysis-result",
            b"voiage.schema_version": result.schema_version.encode("ascii"),
            b"voiage.arrow_schema_version": (
                result.interchange_identity.arrow_schema_version.encode("ascii")
            ),
            b"vop_voiage.contract_version": result.schema_version.encode("ascii"),
            b"vop_voiage.schema_id": b"analysis-result",
            b"vop_voiage.schema_version": (
                result.interchange_identity.arrow_schema_version.encode("ascii")
            ),
            b"vop_voiage.schema_fingerprint": fingerprint.encode("ascii"),
            b"vop_voiage.producer": b"voiage",
            b"vop_voiage.method_contract_version": (
                result.method_contract_version.encode("utf-8")
            ),
            b"vop_voiage.interchange": b"apache-arrow",
        }
    )


def write_analysis_result_ipc[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT], path: Path
) -> None:
    """Write a deterministic Arrow IPC file for one result envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(analysis_result_table(result), b"apache-arrow-ipc")
    with ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)


def write_analysis_result_parquet[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT], path: Path
) -> None:
    """Write a Zstandard-compressed Parquet file for one result envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(
        analysis_result_table(result), b"apache-arrow-parquet"
    )
    pq.write_table(table, path, compression="zstd")


def _write_table_ipc(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(table, b"apache-arrow-ipc")
    with ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)


def _write_table_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _with_interchange_marker(table, b"apache-arrow-parquet")
    pq.write_table(table, path, compression="zstd")


def write_decision_problem_ipc(payload: object, path: Path) -> None:
    """Write a canonical v1 Decision Problem as deterministic Arrow IPC."""
    _write_table_ipc(decision_problem_table(payload), path)


def write_decision_problem_parquet(payload: object, path: Path) -> None:
    """Write a canonical v1 Decision Problem as compressed Parquet."""
    _write_table_parquet(decision_problem_table(payload), path)


def write_statistical_assurance_ipc(payload: object, path: Path) -> None:
    """Write canonical statistical assurance as deterministic Arrow IPC."""
    _write_table_ipc(statistical_assurance_table(payload), path)


def write_statistical_assurance_parquet(payload: object, path: Path) -> None:
    """Write canonical statistical assurance as compressed Parquet."""
    _write_table_parquet(statistical_assurance_table(payload), path)
