"""Arrow interchange for canonical v2 analysis-result envelopes."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

from __future__ import annotations

from hashlib import sha256
import json
from typing import TYPE_CHECKING, TypeVar

import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq

from voiage.contracts.analysis import AnalysisResult, ContractModel

if TYPE_CHECKING:
    from pathlib import Path

PayloadT = TypeVar("PayloadT", bound=ContractModel)


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
        {**(table.schema.metadata or {}), b"vop_voiage.interchange": marker}
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
