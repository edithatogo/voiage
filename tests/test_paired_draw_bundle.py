"""Installed controls for pairing and zero-copy Arrow and Polars handoff."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import polars as pl
import pyarrow as pa
import pytest

from voiage.contracts.normalized_input import NormalizedInputBundle

if TYPE_CHECKING:
    from pathlib import Path

_STRATEGIES = ("intervention_with_long_label", "usual_care")


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "draw_id": [9, 4, 9, 9, 4, 9],
            "strategy_id": [
                _STRATEGIES[1],
                _STRATEGIES[0],
                _STRATEGIES[0],
                _STRATEGIES[0],
                _STRATEGIES[1],
                _STRATEGIES[1],
            ],
            "perspective": ["payer", "payer", "payer", "societal", "payer", "societal"],
            "subgroup_id": ["trial", "trial", "trial", "other", "trial", "other"],
            "net_benefit": [2.0, 7.0, 4.0, 9.0, 2.0, 8.0],
            "draw_weight": [0.4, 0.6, 0.4, 1.0, 0.6, 1.0],
        }
    )


def _load(
    frame: object,
    *,
    strategy_ids: tuple[str, ...] | str = _STRATEGIES,
    weight_field: str | None = "draw_weight",
) -> NormalizedInputBundle:
    experimental = import_module("voiage.experimental")
    return cast(
        "NormalizedInputBundle",
        experimental.paired_draw_bundle(
            frame,
            dataset_id="paired-test",
            strategy_ids=strategy_ids,
            weight_field=weight_field,
        ),
    )


def _addresses(column: pa.ChunkedArray) -> tuple[tuple[int | None, ...], ...]:
    return tuple(
        tuple(
            buffer.address if buffer is not None else None for buffer in chunk.buffers()
        )
        for chunk in column.chunks
    )


def test_arrow_bundle_keeps_buffers_keys_and_binding_across_ipc(tmp_path: Path) -> None:
    arrow = _frame().to_arrow(compat_level=pl.CompatLevel.newest())
    bundle = _load(arrow)
    produced = bundle.table("net_benefits")
    assert produced.equals(arrow)
    for name in arrow.column_names:
        assert _addresses(produced[name]) == _addresses(arrow[name])
    payoff, weight = bundle.manifest.bindings
    assert payoff.strategy_names == _STRATEGIES
    assert payoff.sample_id_field_id == "draw_id"
    assert payoff.strategy_field_id == "strategy_id"
    assert weight.field_ids == ("draw_weight",)
    path = tmp_path / "paired.arrow"
    bundle.write_ipc(path)
    restored = NormalizedInputBundle.read_ipc(path)
    assert restored.table("net_benefits").equals(arrow)
    assert restored.content_digest == bundle.content_digest


def test_polars_bundle_retains_the_declared_flat_column_buffers() -> None:
    frame = _frame()
    arrow = frame.to_arrow(compat_level=pl.CompatLevel.newest())
    produced = _load(frame).table("net_benefits")
    assert produced.equals(arrow)
    for name in produced.column_names:
        assert _addresses(produced[name]) == _addresses(arrow[name])


def test_polars_bundle_preserves_uneven_column_chunk_buffers() -> None:
    draws = pl.concat(
        [pl.Series("draw_id", [9, 4]), pl.Series("draw_id", [9, 9, 4, 9])],
        rechunk=False,
    )
    frame = _frame().with_columns(draws)
    source = pa.chunked_array(
        [
            chunk.to_arrow(compat_level=pl.CompatLevel.newest())
            for chunk in frame["draw_id"].get_chunks()
        ]
    )
    assert source.num_chunks == 2
    produced = _load(frame).table("net_benefits")["draw_id"]
    assert produced.to_pylist() == [9, 4, 9, 9, 4, 9]
    assert _addresses(produced) == _addresses(source)


@pytest.mark.parametrize(
    ("indices", "message"),
    [([0, 0, 1, 2, 3, 4, 5], "duplicate"), ([0, 1, 3, 4, 5], "every strategy")],
)
def test_paired_draws_reject_incomplete_or_duplicate_strategy_rows(
    indices: list[int], message: str
) -> None:
    arrow = _frame().to_arrow(compat_level=pl.CompatLevel.newest())
    selected = pa.concat_tables([arrow.slice(index, 1) for index in indices])
    with pytest.raises(ValueError, match=message):
        _load(selected)


def test_paired_draws_reject_disagreement_in_weights_and_copying_polars_types() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="same draw weight"):
        _load(frame.with_columns(draw_weight=pl.Series([0.7, 0.6, 0.4, 1.0, 0.6, 1.0])))
    with pytest.raises(ValueError, match="zero-copy"):
        _load(frame.with_columns(pl.col("strategy_id").cast(pl.Categorical)))


def test_text_draws_without_declared_weights_retain_their_scope_and_rows() -> None:
    frame = _frame().with_columns(pl.col("draw_id").cast(pl.String))
    bundle = _load(frame, weight_field=None)
    assert tuple(binding.role for binding in bundle.manifest.bindings) == (
        "net_benefit",
    )
    assert bundle.table("net_benefits").equals(
        frame.to_arrow(compat_level=pl.CompatLevel.newest())
    )


@pytest.mark.parametrize(
    "strategy_ids",
    ["usual_care", (), (" usual_care",), ("usual_care", "usual_care")],
)
def test_invalid_strategy_declarations_fail_before_pairing(
    strategy_ids: tuple[str, ...] | str,
) -> None:
    with pytest.raises(ValueError, match="unique identifiers"):
        _load(_frame(), strategy_ids=strategy_ids)


@pytest.mark.parametrize("weight_field", ["draw_id", " "])
def test_weight_field_must_have_its_own_column(weight_field: str) -> None:
    with pytest.raises(ValueError, match="separate column"):
        _load(_frame(), weight_field=weight_field)


@pytest.mark.parametrize("problem", ["empty", "missing", "duplicate"])
def test_paired_draws_require_nonempty_unambiguous_schema(problem: str) -> None:
    table = _frame().to_arrow()
    if problem == "empty":
        table = table.slice(0, 0)
    elif problem == "missing":
        table = table.drop_columns(["perspective"])
    else:
        table = table.append_column("draw_id", table["draw_id"])
    with pytest.raises(ValueError, match="nonempty, uniquely named columns"):
        _load(table)


def test_record_batch_requires_explicit_table_assembly() -> None:
    batch = _frame().to_arrow().to_batches()[0]
    with pytest.raises(TypeError, match="DataFrame or Arrow Table"):
        _load(batch)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("draw_id", 1.25, "valid draw, perspective, and subgroup IDs"),
        ("perspective", "", "valid draw, perspective, and subgroup IDs"),
        ("subgroup_id", " trial", "valid draw, perspective, and subgroup IDs"),
        ("strategy_id", " ", "declared strategy ID"),
        ("strategy_id", "unlisted", "declared strategy ID"),
        ("net_benefit", float("inf"), "net benefits must be finite"),
        ("net_benefit", True, "net benefits must be finite"),
        ("draw_weight", float("nan"), "weights must be finite and nonnegative"),
        ("draw_weight", -0.25, "weights must be finite and nonnegative"),
    ],
)
def test_malformed_draw_rows_fail_before_constructing_bundle(
    column: str, value: object, message: str
) -> None:
    table = _frame().to_arrow()
    replaced = table.set_column(
        table.schema.get_field_index(column),
        column,
        pa.array([value] * table.num_rows),
    )
    with pytest.raises(ValueError, match=message):
        _load(replaced)
