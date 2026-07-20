#!/usr/bin/env python3
"""Measure Arrow serialization against JSON Lines with a robust relative guard."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from io import BytesIO
import json
from pathlib import Path
from statistics import median
import time

import pyarrow as pa
import pyarrow.parquet as pq


def _table(rows: int) -> pa.Table:
    if rows < 1:
        raise ValueError("rows must be positive")
    repeats = (rows + 3) // 4
    return pa.table(
        {
            "choose_under": (["payer", "payer", "societal", "societal"] * repeats)[
                :rows
            ],
            "evaluate_under": (["payer", "societal", "payer", "societal"] * repeats)[
                :rows
            ],
            "chosen_strategy": (["A", "A", "B", "B"] * repeats)[:rows],
            "target_strategy": (["A", "B", "A", "B"] * repeats)[:rows],
            "directional_current_information_evop": ([0.0, 3.0, 3.0, 0.0] * repeats)[
                :rows
            ],
        }
    )


def _elapsed(operation: Callable[[], bytes], repeats: int) -> tuple[float, int]:
    samples: list[float] = []
    size = 0
    for _ in range(repeats):
        started = time.perf_counter()
        size = len(operation())
        samples.append(time.perf_counter() - started)
    return median(samples), size


def benchmark(rows: int = 20_000, repeats: int = 7) -> dict[str, float | int]:
    """Return median timings and sizes for representative interchange output."""
    table = _table(rows)

    def parquet_bytes() -> bytes:
        sink = BytesIO()
        pq.write_table(table, sink, compression="zstd", version="2.6")
        return sink.getvalue()

    def jsonl_bytes() -> bytes:
        return b"\n".join(
            json.dumps(row, separators=(",", ":")).encode("utf-8")
            for row in table.to_pylist()
        )

    parquet_seconds, parquet_size = _elapsed(parquet_bytes, repeats)
    jsonl_seconds, jsonl_size = _elapsed(jsonl_bytes, repeats)
    return {
        "rows": rows,
        "repeats": repeats,
        "parquet_median_seconds": parquet_seconds,
        "jsonl_median_seconds": jsonl_seconds,
        "parquet_to_jsonl_time_ratio": parquet_seconds / jsonl_seconds,
        "parquet_bytes": parquet_size,
        "jsonl_bytes": jsonl_size,
        "parquet_to_jsonl_size_ratio": parquet_size / jsonl_size,
    }


def main(argv: list[str] | None = None) -> int:
    """Emit JSON metrics and reject only material relative regressions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-time-ratio", type=float, default=2.0)
    parser.add_argument("--max-size-ratio", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    metrics = benchmark(args.rows, args.repeats)
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return int(
        metrics["parquet_to_jsonl_time_ratio"] > args.max_time_ratio
        or metrics["parquet_to_jsonl_size_ratio"] > args.max_size_ratio
    )


if __name__ == "__main__":
    raise SystemExit(main())
