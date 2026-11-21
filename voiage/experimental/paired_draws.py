"""Normalize scoped paired draws while retaining the Arrow column buffers."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, TypeGuard

import polars as pl
import pyarrow as pa

from voiage.contracts.normalized_input import VOIBinding
from voiage.ingestion.dataframe import from_dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from voiage.contracts.normalized_input import NormalizedInputBundle

_TABLE = "net_benefits"
_COLUMNS = ("draw_id", "strategy_id", "perspective", "subgroup_id", "net_benefit")
_POLARS_TYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
        pl.String,
    }
)


def paired_draw_bundle(
    frame: pa.Table | pl.DataFrame,
    *,
    dataset_id: str,
    strategy_ids: Sequence[str],
    weight_field: str | None = None,
) -> NormalizedInputBundle:
    """Bind long net benefits to a declared strategy order within each draw scope.

    The table carries `draw_id`, `strategy_id`, `perspective`, `subgroup_id`, and
    `net_benefit`. Draw identifiers can be integers or text. Every draw within
    a perspective and subgroup must contain all declared strategies once, with
    finite values. A declared weight is shared by all rows for the same draw.

    Arrow tables and flat primitive Polars columns keep their buffers at the
    interchange boundary. The returned bundle retains row order and metadata.
    """
    if (
        isinstance(strategy_ids, str)
        or not strategy_ids
        or not all(_identifier(strategy) for strategy in strategy_ids)
        or len(set(strategy_ids)) != len(strategy_ids)
    ):
        raise ValueError(
            "Strategy IDs must be a nonempty sequence of unique identifiers"
        )
    if weight_field is not None and (
        not _identifier(weight_field) or weight_field in _COLUMNS
    ):
        raise ValueError("The weight field must name a separate column")
    table = _arrow_table(frame)
    columns = _COLUMNS if weight_field is None else (*_COLUMNS, weight_field)
    if (
        not table.num_rows
        or len(set(table.column_names)) != table.num_columns
        or not set(columns).issubset(table.column_names)
    ):
        raise ValueError(
            "Paired draws require nonempty, uniquely named columns for all keys and values"
        )
    _validate_draws(table.select(columns), set(strategy_ids), weight_field)
    bindings = [
        VOIBinding(
            role="net_benefit",
            table_id=_TABLE,
            field_ids=("net_benefit",),
            strategy_names=tuple(strategy_ids),
            layout="long",
            sample_id_field_id="draw_id",
            strategy_field_id="strategy_id",
            value_field_id="net_benefit",
        )
    ]
    if weight_field is not None:
        bindings.append(
            VOIBinding(role="weight", table_id=_TABLE, field_ids=(weight_field,))
        )
    return from_dataframe(
        table,
        dataset_id=dataset_id,
        table_id=_TABLE,
        bindings=tuple(bindings),
        allow_copy=False,
    )


def _arrow_table(frame: pa.Table | pl.DataFrame) -> pa.Table:
    if isinstance(frame, pl.DataFrame):
        if not set(frame.dtypes).issubset(_POLARS_TYPES):
            raise ValueError(
                "Polars zero-copy paired draws require flat primitive column types"
            )
        return frame.to_arrow(compat_level=pl.CompatLevel.newest())
    if isinstance(frame, pa.Table):
        return frame
    raise TypeError("Paired draws require a Polars DataFrame or Arrow Table")


def _identifier(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _draw_identity(row: Mapping[str, object]) -> tuple[str, str, int | str]:
    perspective = row["perspective"]
    subgroup = row["subgroup_id"]
    draw = row["draw_id"]
    if not (
        _identifier(perspective)
        and _identifier(subgroup)
        and (type(draw) is int or _identifier(draw))
    ):
        raise ValueError(
            "Paired draw keys require valid draw, perspective, and subgroup IDs"
        )
    return perspective, subgroup, draw


def _validate_draws(
    table: pa.Table, expected_strategies: set[str], weight_field: str | None
) -> None:
    observed: dict[tuple[str, str, int | str], set[str]] = {}
    weights: dict[tuple[str, str, int | str], int | float] = {}
    for batch in table.to_batches(max_chunksize=65_536):
        for row in batch.to_pylist():
            key = _draw_identity(row)
            strategy = row["strategy_id"]
            if not _identifier(strategy) or strategy not in expected_strategies:
                raise ValueError("Paired draw rows must use a declared strategy ID")
            if not _finite_number(row["net_benefit"]):
                raise ValueError("Paired draw net benefits must be finite numbers")
            strategies = observed.setdefault(key, set())
            if strategy in strategies:
                raise ValueError(
                    "Paired draws contain a duplicate scoped draw and strategy row"
                )
            strategies.add(strategy)
            if weight_field is not None:
                weight = row[weight_field]
                if not _finite_number(weight) or weight < 0:
                    raise ValueError("Draw weights must be finite and nonnegative")
                if weights.setdefault(key, weight) != weight:
                    raise ValueError(
                        "Strategies for a scoped draw must share the same draw weight"
                    )
    if any(strategies != expected_strategies for strategies in observed.values()):
        raise ValueError("Each scoped draw must contain every strategy")
