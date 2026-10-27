"""Polars-backed tabular interchange with the established CSV conventions."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

# The default NA vocabulary of the previous CSV reader is part of the I/O contract.
_CSV_NULLS = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]


def read_csv(source: Path | io.StringIO) -> pl.DataFrame:
    """Read a CSV while retaining full-column inference and legacy header names."""
    if isinstance(source, Path):
        with source.open(encoding="utf-8-sig", newline="") as handle:
            text = handle.read()
    else:
        text = source.getvalue().removeprefix("\ufeff")
    stream = io.StringIO(text)
    normalized = io.StringIO()
    writer = csv.writer(normalized, lineterminator="\r\n")
    header = None
    start = 0
    for row in csv.reader(stream, strict=True):
        end = stream.tell()
        # Check the raw record, so quoted whitespace/newlines and comma-only
        # records remain data while unquoted blank physical lines are skipped.
        if text[start:end].strip():
            if header is None:
                header = row
            writer.writerow(row)
        start = end
    if header is None:
        raise ValueError("No columns to parse from file")
    names = [""] * len(header)
    reserved = {name for name in header if name}
    assigned: set[str] = set()
    # Explicit names retain priority over generated names, regardless of order.
    ordered = [index for index, name in enumerate(header) if name]
    ordered.extend(index for index, name in enumerate(header) if not name)
    for index in ordered:
        name = header[index] or f"Unnamed: {index}"
        candidate = name
        suffix = 1
        while candidate in assigned or (suffix > 1 and candidate in reserved):
            candidate = f"{name}.{suffix}"
            suffix += 1
        names[index] = candidate
        assigned.add(candidate)
    frame = pl.read_csv(
        io.StringIO(normalized.getvalue()),
        null_values=_CSV_NULLS,
        infer_schema_length=None,
        new_columns=names,
        try_parse_dates=False,
    )
    # The former reader promotes nullable integer columns to floating point.
    return frame.with_columns(
        pl.col(name).cast(pl.Float64)
        for name, dtype in frame.schema.items()
        if dtype.is_integer() and frame[name].null_count()
    )


def read_parquet(path: Path) -> pl.DataFrame:
    """Read physical columns, excluding index fields recorded by pandas producers."""
    frame = pl.read_parquet(path)
    metadata = pq.read_metadata(path).metadata or {}
    pandas_metadata = json.loads(metadata.get(b"pandas", b"{}"))
    index_columns = [
        column
        for column in pandas_metadata.get("index_columns", [])
        if isinstance(column, str) and column in frame.columns
    ]
    return frame.drop(index_columns)


def records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Retain legacy NaN values for missing scalars in row-oriented payloads."""
    return [
        {name: float("nan") if value is None else value for name, value in row.items()}
        for row in frame.to_dicts()
    ]


def write_csv(frame: pl.DataFrame, path: Path) -> None:
    """Write the established Python float spellings and newline/quoting policy."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(frame.columns)
        for row in frame.iter_rows():
            writer.writerow(
                [
                    ""
                    if value is None or (isinstance(value, float) and math.isnan(value))
                    else value
                    for value in row
                ]
            )
