"""Arrow/Parquet interchange tests for voiage.simulation-outcome.v1 (O06)."""

from __future__ import annotations

import json
from pathlib import Path

from pyarrow import ipc
import pyarrow.parquet as pq
import pytest

from voiage.contracts.interchange import (
    parse_simulation_outcome_table,
    read_simulation_outcome_ipc,
    read_simulation_outcome_parquet,
    simulation_outcome_table,
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
