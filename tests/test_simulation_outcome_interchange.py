"""Arrow/Parquet interchange tests for voiage.simulation-outcome.v1 (O06)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pyarrow as pa
from pyarrow import ipc
import pyarrow.parquet as pq
import pytest

from voiage.contracts import normalized_input
from voiage.contracts.interchange import (
    SIMULATION_OUTCOME_PROFILE_VERSION_V1,
    SimulationOutcomeProfileMetadata,
    SimulationOutcomeProvenance,
    SimulationOutcomeValueDeclaration,
    _as_finite_float,
    _validate_draw_weights,
    parse_simulation_outcome_table,
    read_simulation_outcome_ipc,
    read_simulation_outcome_parquet,
    schema_fingerprint,
    simulation_outcome_table,
    validate_untrimmed_nonempty,
    write_simulation_outcome_ipc,
    write_simulation_outcome_parquet,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "specs"
    / "ecosystem"
    / "simulation-outcome"
    / "v1"
    / "interchange-fixture.json"
)


def _fixtures() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _keyed_outcomes(
    profile: dict[str, object],
) -> dict[tuple[str, str], tuple[float, float]]:
    outcomes = profile["outcomes"]
    assert isinstance(outcomes, list)
    return {
        (row["draw_key"], row["strategy_id"]): (
            float(row["value"]),
            float(row["weight"]),
        )
        for row in outcomes
    }


@pytest.mark.parametrize(
    "fixture_name",
    ["synthetic_health_with_parameters", "synthetic_non_health_shuffled"],
)
def test_simulation_outcome_ipc_and_parquet_preserve_semantics_across_chunks(
    tmp_path: Path, fixture_name: str
) -> None:
    profile = _fixtures()[fixture_name]
    assert isinstance(profile, dict)

    table = simulation_outcome_table(profile, chunk_size=2)
    assert table.schema.metadata[b"voiage.contract"] == b"simulation-outcome"
    assert (
        table.schema.metadata[b"voiage.simulation_outcome.profile_version"]
        == b"voiage.simulation-outcome.v1"
    )
    assert any(column.num_chunks >= 2 for column in table.columns)

    ipc_path = tmp_path / f"{fixture_name}.arrow"
    parquet_path = tmp_path / f"{fixture_name}.parquet"
    write_simulation_outcome_ipc(profile, ipc_path, chunk_size=2)
    write_simulation_outcome_parquet(profile, parquet_path, chunk_size=2)

    from_ipc = read_simulation_outcome_ipc(ipc_path)
    from_parquet = read_simulation_outcome_parquet(parquet_path)

    for restored in (parse_simulation_outcome_table(table), from_ipc, from_parquet):
        assert restored["profile_version"] == profile["profile_version"]
        assert restored["provenance"] == profile["provenance"]
        assert restored["draw_namespace"] == profile["draw_namespace"]
        assert restored["strategy_ids"] == profile["strategy_ids"]
        assert restored["value_declaration"] == profile["value_declaration"]
        assert restored["weight_semantics"] == profile["weight_semantics"]
        assert "draw_keys" not in restored
        assert _keyed_outcomes(restored) == _keyed_outcomes(profile)
        expected_params = profile.get("parameter_samples")
        if expected_params is None:
            assert restored["parameter_samples"] is None
        else:
            expected_by_key = {
                row["draw_key"]: row["values"] for row in expected_params
            }
            restored_by_key = {
                row["draw_key"]: row["values"] for row in restored["parameter_samples"]
            }
            assert restored_by_key == expected_by_key


@pytest.mark.parametrize(
    "missing_meta_key",
    [
        b"voiage.contract",
        b"voiage.simulation_outcome.profile_version",
        b"voiage.simulation_outcome.metadata",
    ],
)
def test_dropped_required_metadata_fails_closed(
    tmp_path: Path, missing_meta_key: bytes
) -> None:
    profile = _fixtures()["synthetic_health_with_parameters"]
    assert isinstance(profile, dict)
    table = simulation_outcome_table(profile)
    metadata = dict(table.schema.metadata or {})
    del metadata[missing_meta_key]
    damaged = table.replace_schema_metadata(metadata)

    with pytest.raises(ValueError):
        parse_simulation_outcome_table(damaged)

    ipc_path = tmp_path / "damaged.arrow"
    with ipc.new_file(ipc_path, damaged.schema) as writer:
        writer.write_table(damaged)
    with pytest.raises(ValueError):
        read_simulation_outcome_ipc(ipc_path)

    parquet_path = tmp_path / "damaged.parquet"
    pq.write_table(damaged, parquet_path, compression="zstd")
    with pytest.raises(ValueError):
        read_simulation_outcome_parquet(parquet_path)


def test_dropped_required_column_fails_closed(tmp_path: Path) -> None:
    profile = _fixtures()["synthetic_non_health_shuffled"]
    assert isinstance(profile, dict)
    table = simulation_outcome_table(profile)
    damaged = table.drop_columns(["weight"])

    with pytest.raises(ValueError):
        parse_simulation_outcome_table(damaged)

    parquet_path = tmp_path / "no-weight.parquet"
    pq.write_table(damaged, parquet_path, compression="zstd")
    with pytest.raises(ValueError):
        read_simulation_outcome_parquet(parquet_path)


def test_mismatched_arrow_column_types_fail_closed() -> None:
    import pyarrow as pa

    from voiage.contracts.interchange import schema_fingerprint

    profile = _fixtures()["synthetic_health_with_parameters"]
    assert isinstance(profile, dict)
    table = simulation_outcome_table(profile)

    int_value_table = table.set_column(
        table.column_names.index("value"),
        "value",
        pa.array([1] * table.num_rows, type=pa.int64()),
    )
    meta_int = dict(int_value_table.schema.metadata or {})
    meta_int[b"voiage.schema_fingerprint"] = schema_fingerprint(
        int_value_table.schema
    ).encode("ascii")
    with pytest.raises(ValueError, match="float64 Arrow type"):
        parse_simulation_outcome_table(
            int_value_table.replace_schema_metadata(meta_int)
        )

    int_key_table = table.set_column(
        table.column_names.index("draw_key"),
        "draw_key",
        pa.array([1] * table.num_rows, type=pa.int64()),
    )
    meta_str = dict(int_key_table.schema.metadata or {})
    meta_str[b"voiage.schema_fingerprint"] = schema_fingerprint(
        int_key_table.schema
    ).encode("ascii")
    with pytest.raises(ValueError, match="string Arrow type"):
        parse_simulation_outcome_table(int_key_table.replace_schema_metadata(meta_str))


def test_reexports_and_model_validation_boundaries() -> None:
    assert (
        normalized_input.SIMULATION_OUTCOME_PROFILE_VERSION_V1
        == SIMULATION_OUTCOME_PROFILE_VERSION_V1
    )
    assert normalized_input.validate_untrimmed_nonempty is validate_untrimmed_nonempty
    assert normalized_input.SimulationOutcomeProvenance is SimulationOutcomeProvenance
    assert (
        normalized_input.SimulationOutcomeValueDeclaration
        is SimulationOutcomeValueDeclaration
    )
    assert (
        normalized_input.SimulationOutcomeProfileMetadata
        is SimulationOutcomeProfileMetadata
    )

    with pytest.raises(ValueError, match="must not be empty"):
        validate_untrimmed_nonempty("   ", "unit")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        validate_untrimmed_nonempty(" QALY ", "unit")

    provenance = SimulationOutcomeProvenance(
        producer_id="prod",
        producer_version="1.0",
        model_revision="rev1",
        valuation_source="val",
        artifact_digest="digest",
    )
    valid_decl = SimulationOutcomeValueDeclaration(
        unit="QALY",
        perspective="healthcare",
        convention="net_benefit",
        objective="maximize",
        currency_price_year="USD-2026",
        population_basis="cohort",
        horizon="lifetime",
        discounting_status="3.5%",
    )
    assert valid_decl.population_basis == "cohort"

    with pytest.raises(ValueError, match="incompatible value convention"):
        SimulationOutcomeValueDeclaration(
            unit="QALY",
            perspective="healthcare",
            convention="loss",
            objective="maximize",
        )

    with pytest.raises(ValueError, match="strategy_ids must not be empty"):
        SimulationOutcomeProfileMetadata(
            profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
            provenance=provenance,
            draw_namespace="ns",
            strategy_ids=(),
            value_declaration=valid_decl,
            weight_semantics="equal",
            draw_keys=("d1",),
        )
    with pytest.raises(ValueError, match="strategy_ids must be unique"):
        SimulationOutcomeProfileMetadata(
            profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
            provenance=provenance,
            draw_namespace="ns",
            strategy_ids=("a", "a"),
            value_declaration=valid_decl,
            weight_semantics="equal",
            draw_keys=("d1",),
        )
    with pytest.raises(ValueError, match="draw_keys must not be empty"):
        SimulationOutcomeProfileMetadata(
            profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
            provenance=provenance,
            draw_namespace="ns",
            strategy_ids=("a",),
            value_declaration=valid_decl,
            weight_semantics="equal",
            draw_keys=(),
        )
    with pytest.raises(ValueError, match="draw_keys must be unique"):
        SimulationOutcomeProfileMetadata(
            profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
            provenance=provenance,
            draw_namespace="ns",
            strategy_ids=("a",),
            value_declaration=valid_decl,
            weight_semantics="equal",
            draw_keys=("d1", "d1"),
        )
    with pytest.raises(ValueError, match="parameter_names must not be empty"):
        SimulationOutcomeProfileMetadata(
            profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
            provenance=provenance,
            draw_namespace="ns",
            strategy_ids=("a",),
            value_declaration=valid_decl,
            weight_semantics="equal",
            draw_keys=("d1",),
            parameter_names=(),
        )
    with pytest.raises(ValueError, match="parameter_names must be unique"):
        SimulationOutcomeProfileMetadata(
            profile_version=SIMULATION_OUTCOME_PROFILE_VERSION_V1,
            provenance=provenance,
            draw_namespace="ns",
            strategy_ids=("a",),
            value_declaration=valid_decl,
            weight_semantics="equal",
            draw_keys=("d1",),
            parameter_names=("theta", "theta"),
        )


def test_scalar_and_weight_validation_edges() -> None:
    assert _as_finite_float(-0.0, "x") == 0.0
    assert _as_finite_float(2, "x") == 2.0
    with pytest.raises(TypeError, match="finite number"):
        _as_finite_float(True, "x")
    with pytest.raises(TypeError, match="finite number"):
        _as_finite_float("1.0", "x")
    with pytest.raises(ValueError, match="must be finite"):
        _as_finite_float(math.nan, "x")

    with pytest.raises(ValueError, match="outcomes must not be empty"):
        _validate_draw_weights([], "equal")
    _validate_draw_weights([0.25, 0.25], "equal")
    with pytest.raises(ValueError, match="identical positive weights"):
        _validate_draw_weights([0.0, 0.0], "equal")
    with pytest.raises(ValueError, match="identical positive weights"):
        _validate_draw_weights([0.25, 0.5], "equal")
    with pytest.raises(ValueError, match="sum to 1.0"):
        _validate_draw_weights([0.4, 0.4], "probability")
    _validate_draw_weights([2.0, 3.0], "relative_weight")
    with pytest.raises(ValueError, match="finite positive sum"):
        _validate_draw_weights([0.0, 0.0], "relative_weight")
    with pytest.raises(ValueError, match="unsupported weight semantics"):
        _validate_draw_weights([1.0], "bogus")


def test_simulation_outcome_table_rejects_malformed_inputs() -> None:
    base = _fixtures()["synthetic_health_with_parameters"]
    assert isinstance(base, dict)

    bad_version = copy.deepcopy(base)
    bad_version["profile_version"] = "voiage.simulation-outcome.v99"
    with pytest.raises(ValueError, match="unsupported simulation outcome"):
        simulation_outcome_table(bad_version)

    bad_strats = copy.deepcopy(base)
    bad_strats["strategy_ids"] = "soc"
    with pytest.raises(TypeError, match="strategy_ids must be a sequence"):
        simulation_outcome_table(bad_strats)

    bad_ns = copy.deepcopy(base)
    bad_ns["draw_namespace"] = 123
    with pytest.raises(TypeError, match="draw_namespace must be a string"):
        simulation_outcome_table(bad_ns)

    bad_ws = copy.deepcopy(base)
    bad_ws["weight_semantics"] = "bogus"
    with pytest.raises(ValueError, match="unsupported weight_semantics"):
        simulation_outcome_table(bad_ws)

    rel = copy.deepcopy(base)
    rel["weight_semantics"] = "relative_weight"
    for row in rel["outcomes"]:  # type: ignore[union-attr]
        row["weight"] = 2.0
    assert simulation_outcome_table(rel, chunk_size=1).num_rows == len(rel["outcomes"])

    eq = copy.deepcopy(base)
    eq["weight_semantics"] = "equal"
    for row in eq["outcomes"]:  # type: ignore[union-attr]
        row["weight"] = 1.0
    assert (
        parse_simulation_outcome_table(simulation_outcome_table(eq))["weight_semantics"]
        == "equal"
    )

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        simulation_outcome_table(base, chunk_size=0)

    bad_outcomes_type = copy.deepcopy(base)
    bad_outcomes_type["outcomes"] = "not-a-seq"
    with pytest.raises(TypeError, match="outcomes must be a non-empty sequence"):
        simulation_outcome_table(bad_outcomes_type)

    empty_outcomes = copy.deepcopy(base)
    empty_outcomes["outcomes"] = []
    with pytest.raises(ValueError, match="outcomes must not be empty"):
        simulation_outcome_table(empty_outcomes)

    non_map_row = copy.deepcopy(base)
    non_map_row["outcomes"] = ["bad-row"]
    with pytest.raises(TypeError, match="outcome record must be a mapping"):
        simulation_outcome_table(non_map_row)

    wrong_row_ns = copy.deepcopy(base)
    wrong_row_ns["outcomes"][0]["draw_namespace"] = "other-ns"  # type: ignore[index]
    with pytest.raises(ValueError, match="draw_namespace mismatch"):
        simulation_outcome_table(wrong_row_ns)

    unknown_strat = copy.deepcopy(base)
    unknown_strat["outcomes"][0]["strategy_id"] = "unknown"  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown strategy_id"):
        simulation_outcome_table(unknown_strat)

    neg_weight = copy.deepcopy(base)
    neg_weight["outcomes"][0]["weight"] = -1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="negative weight"):
        simulation_outcome_table(neg_weight)

    decl_match = copy.deepcopy(base)
    decl_match["outcomes"][0]["declaration"] = copy.deepcopy(
        decl_match["value_declaration"]
    )  # type: ignore[index]
    assert simulation_outcome_table(decl_match).num_rows == len(base["outcomes"])

    decl_conflict = copy.deepcopy(base)
    conflict_decl = copy.deepcopy(decl_conflict["value_declaration"])
    conflict_decl["unit"] = "DALY"  # type: ignore[index]
    decl_conflict["outcomes"][0]["declaration"] = conflict_decl  # type: ignore[index]
    with pytest.raises(ValueError, match="conflicts with profile declaration"):
        simulation_outcome_table(decl_conflict)

    weight_conflict = copy.deepcopy(base)
    for row in weight_conflict["outcomes"]:  # type: ignore[union-attr]
        if (
            row["draw_key"] == base["outcomes"][0]["draw_key"]
            and row["strategy_id"] != base["outcomes"][0]["strategy_id"]
        ):  # type: ignore[index]
            row["weight"] = row["weight"] + 0.1
            break
    with pytest.raises(ValueError, match="conflicting draw weight"):
        simulation_outcome_table(weight_conflict)

    dup_row = copy.deepcopy(base)
    dup_row["outcomes"].append(copy.deepcopy(dup_row["outcomes"][0]))  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="duplicate outcome record"):
        simulation_outcome_table(dup_row)

    incomplete = copy.deepcopy(base)
    incomplete["outcomes"] = incomplete["outcomes"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="incomplete draw"):
        simulation_outcome_table(incomplete)


def test_parameter_samples_validation_edges() -> None:
    base = _fixtures()["synthetic_health_with_parameters"]
    assert isinstance(base, dict)

    bad_seq = copy.deepcopy(base)
    bad_seq["parameter_samples"] = "bad"
    with pytest.raises(TypeError, match="parameter_samples must be a sequence"):
        simulation_outcome_table(bad_seq)

    wrong_ns = copy.deepcopy(base)
    wrong_ns["parameter_samples"][0]["draw_namespace"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="parameter sample draw_namespace mismatch"):
        simulation_outcome_table(wrong_ns)

    dup_draw = copy.deepcopy(base)
    dup_draw["parameter_samples"].append(
        copy.deepcopy(dup_draw["parameter_samples"][0])
    )  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="duplicate parameter draw"):
        simulation_outcome_table(dup_draw)

    unmatched = copy.deepcopy(base)
    unmatched["parameter_samples"][0]["draw_key"] = "unknown-draw"  # type: ignore[index]
    with pytest.raises(ValueError, match="unmatched parameter draw"):
        simulation_outcome_table(unmatched)

    empty_vals = copy.deepcopy(base)
    empty_vals["parameter_samples"][0]["values"] = {}  # type: ignore[index]
    with pytest.raises(ValueError, match="empty parameter values"):
        simulation_outcome_table(empty_vals)

    inconsistent = copy.deepcopy(base)
    inconsistent["parameter_samples"][1]["values"] = {"other_param": 1.0}  # type: ignore[index]
    with pytest.raises(ValueError, match="inconsistent parameter schema"):
        simulation_outcome_table(inconsistent)

    missing_draw = copy.deepcopy(base)
    missing_draw["parameter_samples"] = missing_draw["parameter_samples"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing parameter draw"):
        simulation_outcome_table(missing_draw)


def test_parse_and_file_interchange_failure_modes(tmp_path: Path) -> None:
    base = _fixtures()["synthetic_health_with_parameters"]
    assert isinstance(base, dict)
    table = simulation_outcome_table(base)
    meta = dict(table.schema.metadata or {})

    # Invalid contract marker.
    bad_contract = table.replace_schema_metadata({**meta, b"voiage.contract": b"other"})
    with pytest.raises(ValueError, match="invalid simulation-outcome contract marker"):
        parse_simulation_outcome_table(bad_contract)

    # Unsupported profile_version bytes.
    bad_version = table.replace_schema_metadata(
        {
            **meta,
            b"voiage.simulation_outcome.profile_version": b"voiage.simulation-outcome.v99",
        }
    )
    with pytest.raises(
        ValueError, match="unsupported simulation-outcome profile_version"
    ):
        parse_simulation_outcome_table(bad_version)

    # Corrupt metadata JSON.
    bad_meta_json = table.replace_schema_metadata(
        {**meta, b"voiage.simulation_outcome.metadata": b"{not-valid"}
    )
    with pytest.raises(ValueError, match="invalid simulation-outcome metadata header"):
        parse_simulation_outcome_table(bad_meta_json)

    # Metadata schema mismatch against table columns (while fingerprint matches Arrow schema).
    parsed_header = json.loads(
        meta[b"voiage.simulation_outcome.metadata"].decode("utf-8")
    )
    extra_param_header = copy.deepcopy(parsed_header)
    extra_param_header["parameter_names"] = ["extra_col"]
    col_mismatch = table.replace_schema_metadata(
        {
            **meta,
            b"voiage.simulation_outcome.metadata": json.dumps(
                extra_param_header
            ).encode("utf-8"),
        }
    )
    with pytest.raises(ValueError, match="do not match metadata schema"):
        parse_simulation_outcome_table(col_mismatch)

    # Metadata draw_keys mismatch against rows (while fingerprint & columns match).
    extra_draw_header = copy.deepcopy(parsed_header)
    extra_draw_header["draw_keys"] = [*parsed_header["draw_keys"], "extra-draw"]
    draw_mismatch = table.replace_schema_metadata(
        {
            **meta,
            b"voiage.simulation_outcome.metadata": json.dumps(extra_draw_header).encode(
                "utf-8"
            ),
        }
    )
    with pytest.raises(ValueError, match="draw_keys do not match metadata header"):
        parse_simulation_outcome_table(draw_mismatch)

    # Inconsistent repeated parameter values across strategies for one draw_key.
    param_col_name = next(
        name for name in table.column_names if name.startswith("param::")
    )
    rows = table.to_pylist()
    target_draw = rows[0]["draw_key"]
    for idx, row in enumerate(rows):
        if idx > 0 and row["draw_key"] == target_draw:
            row[param_col_name] = float(row[param_col_name]) + 42.0
            break
    tampered_table = pa.Table.from_pylist(
        rows, schema=table.schema.remove_metadata()
    ).replace_schema_metadata(
        {
            **meta,
            b"vop_voiage.schema_fingerprint": schema_fingerprint(table.schema).encode(
                "ascii"
            ),
        }
    )
    with pytest.raises(ValueError, match="inconsistent repeated parameter values"):
        parse_simulation_outcome_table(tampered_table)

    # Unchunked IPC write + container marker mismatch checks.
    unchunked_ipc = tmp_path / "unchunked.arrow"
    write_simulation_outcome_ipc(base, unchunked_ipc, chunk_size=None)
    assert len(read_simulation_outcome_ipc(unchunked_ipc)["outcomes"]) == len(
        base["outcomes"]
    )

    parquet_path = tmp_path / "good.parquet"
    write_simulation_outcome_parquet(base, parquet_path, chunk_size=None)
    assert len(read_simulation_outcome_parquet(parquet_path)["outcomes"]) == len(
        base["outcomes"]
    )
    wrong_ipc_path = _write_raw_ipc_with_marker(
        tmp_path / "wrong-marker.arrow", table, b"apache-arrow-parquet"
    )
    with pytest.raises(
        ValueError, match="not a voiage simulation-outcome Arrow IPC file"
    ):
        read_simulation_outcome_ipc(wrong_ipc_path)
    wrong_parquet_path = _write_raw_parquet_with_marker(
        tmp_path / "wrong-marker.parquet", table, b"apache-arrow-ipc"
    )
    with pytest.raises(
        ValueError, match="not a voiage simulation-outcome Parquet file"
    ):
        read_simulation_outcome_parquet(wrong_parquet_path)


def _write_raw_ipc_with_marker(path: Path, table: pa.Table, marker: bytes) -> Path:
    marked = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"vop_voiage.interchange": marker}
    )
    with ipc.new_file(path, marked.schema) as writer:
        writer.write_table(marked)
    return path


def _write_raw_parquet_with_marker(path: Path, table: pa.Table, marker: bytes) -> Path:
    marked = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"vop_voiage.interchange": marker}
    )
    pq.write_table(marked, path, compression="zstd")
    return path
