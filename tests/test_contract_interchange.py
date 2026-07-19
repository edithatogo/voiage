"""Arrow/Parquet/IPC conformance for canonical v2 result envelopes."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from pyarrow import ipc
import pyarrow.parquet as pq

from voiage.contracts.analysis import AnalysisSpec, NumericalPolicy
from voiage.contracts.interchange import (
    analysis_result_table,
    write_analysis_result_ipc,
    write_analysis_result_parquet,
)
from voiage.contracts.kernel import run_evpi


def _result():
    policy = NumericalPolicy(backend_preference=("numpy",))
    spec = AnalysisSpec(
        analysis_id="arrow-v2",
        decision_problem_id="decision",
        method_family="evpi",
        method_contract_version="1.0.0",
        strategy_names=("A", "B"),
        numerical_policy=policy,
    )
    return run_evpi(np.array([[1.0, 2.0], [3.0, 1.0]]), spec=spec)


def test_arrow_table_has_stable_v2_identity_and_json_payload() -> None:
    result = _result()
    table = analysis_result_table(result)
    assert table.schema.metadata[b"voiage.contract"] == b"analysis-result"
    assert table.schema.metadata[b"voiage.schema_version"] == b"2.0.0"
    assert json.loads(table["result_json"][0].as_py()) == result.model_dump(mode="json")


def test_ipc_and_parquet_round_trip_through_pyarrow_and_polars(tmp_path) -> None:
    result = _result()
    ipc_path = tmp_path / "result.arrow"
    parquet_path = tmp_path / "result.parquet"
    write_analysis_result_ipc(result, ipc_path)
    write_analysis_result_parquet(result, parquet_path)
    with ipc.open_file(ipc_path) as reader:
        ipc_table = reader.read_all()
    parquet_table = pq.read_table(parquet_path)
    assert ipc_table.to_pylist() == parquet_table.to_pylist()
    assert pl.read_ipc(ipc_path).to_dicts() == pl.read_parquet(parquet_path).to_dicts()

    second_ipc = tmp_path / "result-second.arrow"
    second_parquet = tmp_path / "result-second.parquet"
    write_analysis_result_ipc(result, second_ipc)
    write_analysis_result_parquet(result, second_parquet)
    assert second_ipc.read_bytes() == ipc_path.read_bytes()
    assert second_parquet.read_bytes() == parquet_path.read_bytes()
