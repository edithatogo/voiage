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
    text = (
        source.read_text(encoding="utf-8-sig")
        if isinstance(source, Path)
        else source.getvalue()
    )
    rows = csv.reader(io.StringIO(text))
    header = next((row for row in rows if row), None)
    if header is None:
        raise ValueError("No columns to parse from file")
    names: list[str] = []
    reserved = {name for name in header if name}
    for index, name in enumerate(header):
        candidate = name or f"Unnamed: {index}"
        suffix = 1
        while candidate in names or (suffix > 1 and candidate in reserved):
            candidate = f"{name}.{suffix}"
            suffix += 1
        names.append(candidate)
    frame = pl.read_csv(
        io.StringIO(text),
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
