"""Arrow/Parquet/IPC conformance for canonical v2 result envelopes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from pyarrow import ipc
import pyarrow.parquet as pq
import pytest

from voiage import _runtime
from voiage.contracts.analysis import AnalysisSpec, NumericalPolicy
from voiage.contracts.interchange import (
    analysis_result_table,
    decision_problem_table,
    schema_fingerprint,
    statistical_assurance_table,
    write_analysis_result_ipc,
    write_analysis_result_parquet,
    write_decision_problem_ipc,
    write_decision_problem_parquet,
    write_statistical_assurance_ipc,
    write_statistical_assurance_parquet,
)
from voiage.contracts.kernel import run_evpi
from voiage.exceptions import SerializationError


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
    assert table.schema.metadata[b"vop_voiage.contract_version"] == b"2.0.0"
    assert table.schema.metadata[b"vop_voiage.schema_id"] == b"analysis-result"
    assert table.schema.metadata[b"vop_voiage.schema_version"] == b"1.0.0"
    assert table.schema.metadata[b"vop_voiage.producer"] == b"voiage"
    assert table.schema.metadata[b"vop_voiage.interchange"] == b"apache-arrow"
    assert table.schema.metadata[b"vop_voiage.method_contract_version"] == b"1.0.0"
    assert table.schema.metadata[b"vop_voiage.schema_fingerprint"].decode() == (
        schema_fingerprint(table.schema)
    )
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
    assert ipc_table.schema.metadata[b"vop_voiage.interchange"] == (b"apache-arrow-ipc")
    assert parquet_table.schema.metadata[b"vop_voiage.interchange"] == (
        b"apache-arrow-parquet"
    )
    assert schema_fingerprint(ipc_table.schema) == schema_fingerprint(
        parquet_table.schema
    )
    assert ipc_table.to_pylist() == parquet_table.to_pylist()
    assert pl.read_ipc(ipc_path).to_dicts() == pl.read_parquet(parquet_path).to_dicts()

    second_ipc = tmp_path / "result-second.arrow"
    second_parquet = tmp_path / "result-second.parquet"
    write_analysis_result_ipc(result, second_ipc)
    write_analysis_result_parquet(result, second_parquet)
    assert second_ipc.read_bytes() == ipc_path.read_bytes()
    assert second_parquet.read_bytes() == parquet_path.read_bytes()


def test_schema_fingerprint_excludes_container_metadata() -> None:
    table = analysis_result_table(_result())
    changed = table.replace_schema_metadata({b"container": b"different"})
    assert schema_fingerprint(table.schema) == schema_fingerprint(changed.schema)


ROOT = Path(__file__).resolve().parents[1]
DECISION_PROBLEM = ROOT / "specs/core-api/examples/v1/decision-problem.example.json"
STATISTICAL_ASSURANCE = (
    ROOT / "specs/core-api/examples/v1/statistical-assurance.example.json"
)


@pytest.mark.parametrize(
    (
        "contract",
        "source",
        "normalizer",
        "table_factory",
        "ipc_writer",
        "parquet_writer",
    ),
    [
        (
            "decision-problem",
            DECISION_PROBLEM,
            _runtime.normalize_decision_problem_json,
            decision_problem_table,
            write_decision_problem_ipc,
            write_decision_problem_parquet,
        ),
        (
            "statistical-assurance",
            STATISTICAL_ASSURANCE,
            _runtime.normalize_statistical_assurance_json,
            statistical_assurance_table,
            write_statistical_assurance_ipc,
            write_statistical_assurance_parquet,
        ),
    ],
)
def test_v1_contract_arrow_round_trip_is_lossless_and_deterministic(
    tmp_path,
    contract,
    source,
    normalizer,
    table_factory,
    ipc_writer,
    parquet_writer,
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    table = table_factory(payload)
    metadata = table.schema.metadata
    assert metadata[b"voiage.contract"] == contract.encode()
    assert metadata[b"voiage.schema_version"] == b"1.0.0"
    assert metadata[b"voiage.arrow_schema_version"] == b"1.0.0"
    assert metadata[b"voiage.producer"] == b"voiage-rust-validated"
    assert metadata[b"voiage.schema_fingerprint"].decode() == schema_fingerprint(
        table.schema
    )
    assert json.loads(table["payload_json"][0].as_py()) == normalizer(payload)

    ipc_path = tmp_path / f"{contract}.arrow"
    parquet_path = tmp_path / f"{contract}.parquet"
    ipc_writer(payload, ipc_path)
    parquet_writer(payload, parquet_path)
    with ipc.open_file(ipc_path) as reader:
        ipc_table = reader.read_all()
    parquet_table = pq.read_table(parquet_path)
    assert ipc_table.to_pylist() == parquet_table.to_pylist()
    assert pl.read_ipc(ipc_path).to_dicts() == pl.read_parquet(parquet_path).to_dicts()
    assert ipc_table.schema.metadata[b"voiage.interchange"] == b"apache-arrow-ipc"
    assert parquet_table.schema.metadata[b"voiage.interchange"] == (
        b"apache-arrow-parquet"
    )

    second_ipc = tmp_path / f"{contract}-second.arrow"
    second_parquet = tmp_path / f"{contract}-second.parquet"
    ipc_writer(payload, second_ipc)
    parquet_writer(payload, second_parquet)
    assert second_ipc.read_bytes() == ipc_path.read_bytes()
    assert second_parquet.read_bytes() == parquet_path.read_bytes()


def test_contract_arrow_rejects_invalid_payload_before_serialization() -> None:
    decision = json.loads(DECISION_PROBLEM.read_text(encoding="utf-8"))
    decision["willingness_to_pay"] = 0
    with pytest.raises(SerializationError):
        decision_problem_table(decision)

    assurance = json.loads(STATISTICAL_ASSURANCE.read_text(encoding="utf-8"))
    assurance["confidence_interval"]["lower"] = 2.0
    assurance["confidence_interval"]["upper"] = 1.0
    with pytest.raises(SerializationError):
        statistical_assurance_table(assurance)


@pytest.mark.parametrize(
    ("descriptor_name", "source", "table_factory"),
    [
        (
            "decision-problem.schema.json",
            DECISION_PROBLEM,
            decision_problem_table,
        ),
        (
            "statistical-assurance.schema.json",
            STATISTICAL_ASSURANCE,
            statistical_assurance_table,
        ),
    ],
)
def test_v1_arrow_schema_matches_language_neutral_descriptor(
    descriptor_name,
    source,
    table_factory,
) -> None:
    descriptor_path = ROOT / "specs/core-api/arrow/v1" / descriptor_name
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload = json.loads(source.read_text(encoding="utf-8"))
    table = table_factory(payload)
    fields = [
        {
            "name": field.name,
            "arrow_type": str(field.type),
            "nullable": field.nullable,
        }
        for field in table.schema
    ]
    assert fields == descriptor["fields"]
    assert schema_fingerprint(table.schema) == descriptor["logical_schema_fingerprint"]
