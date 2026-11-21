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


def _load(frame: pl.DataFrame | pa.Table) -> NormalizedInputBundle:
    experimental = import_module("voiage.experimental")
    return cast(
        "NormalizedInputBundle",
        experimental.paired_draw_bundle(
            frame,
            dataset_id="paired-test",
            strategy_ids=_STRATEGIES,
            weight_field="draw_weight",
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
