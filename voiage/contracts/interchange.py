"""Arrow interchange for canonical v2 analysis-result envelopes."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq

from voiage.contracts.analysis import AnalysisResult, ContractModel

if TYPE_CHECKING:
    from pathlib import Path


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
    return table.replace_schema_metadata(
        {
            b"voiage.contract": b"analysis-result",
            b"voiage.schema_version": result.schema_version.encode("ascii"),
            b"voiage.arrow_schema_version": (
                result.interchange_identity.arrow_schema_version.encode("ascii")
            ),
        }
    )


def write_analysis_result_ipc[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT], path: Path
) -> None:
    """Write a deterministic Arrow IPC file for one result envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = analysis_result_table(result)
    with ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)


def write_analysis_result_parquet[PayloadT: ContractModel](
    result: AnalysisResult[PayloadT], path: Path
) -> None:
    """Write a Zstandard-compressed Parquet file for one result envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(analysis_result_table(result), path, compression="zstd")
