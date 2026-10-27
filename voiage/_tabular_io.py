"""Polars-backed tabular interchange with the established CSV conventions."""

from __future__ import annotations

import csv
import io
import json
import math
import os
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
    # CSV has no physical schema. Retain the legacy numeric policy for nullable
    # integers and non-empty columns containing only missing tokens. Mixed text
    # remains String; only columns with no values can change String to Float64.
    return frame.with_columns(
        pl.col(name).cast(pl.Float64)
        for name, dtype in frame.schema.items()
        if frame[name].null_count()
        and (dtype.is_integer() or frame[name].null_count() == frame.height)
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
        writer = csv.writer(stream, lineterminator=os.linesep)
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


def _excel_cell(value: Any) -> Any:
    """Normalize the native reader's scalars to the established cell policy."""
    import datetime as dt

    if isinstance(value, str) and value in _CSV_NULLS:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return dt.datetime.combine(value, dt.time())
    return value


def _excel_headers(values: list[Any]) -> list[Any]:
    """Deduplicate typed headers, giving explicit names priority over blanks."""
    names: list[Any] = [None] * len(values)
    explicit = [i for i, value in enumerate(values) if value not in (None, "")]
    reserved = {values[i] for i in explicit}
    assigned: set[Any] = set()
    for index in explicit + [i for i in range(len(values)) if i not in explicit]:
        name = values[index] if index in explicit else f"Unnamed: {index}"
        candidate = name
        suffix = 1
        while candidate in assigned or (suffix > 1 and candidate in reserved):
            candidate = f"{name}.{suffix}"
            suffix += 1
        names[index] = candidate
        assigned.add(candidate)
    return names


def read_excel(path: Path) -> tuple[pl.DataFrame, list[Any]]:
    """Read raw worksheet cells without cropping blanks or coercing mixed cells."""
    import datetime as dt

    try:
        from python_calamine import CalamineWorkbook, SheetTypeEnum
    except ImportError as error:
        raise ImportError(
            "Excel reading requires python-calamine; install voiage[excel]."
        ) from error
    with CalamineWorkbook.from_path(path) as workbook:
        sheets = [
            s for s in workbook.sheets_metadata if s.typ == SheetTypeEnum.WorkSheet
        ]
        if not sheets:
            raise ValueError("Workbook contains no worksheets")
        rows = workbook.get_sheet_by_name(sheets[0].name).to_python(
            skip_empty_area=False
        )
    if not rows:
        return pl.DataFrame(), []
    headers = _excel_headers(
        [int(v) if isinstance(v, float) and v.is_integer() else v for v in rows[0]]
    )
    names = [f"column_{i}" for i in range(len(headers))]
    columns = [[_excel_cell(row[i]) for row in rows[1:]] for i in range(len(headers))]
    bool_tokens = {"True", "true", "TRUE", "False", "false", "FALSE"}
    bool_columns = [
        all(
            isinstance(v, bool) or (isinstance(v, str) and v in bool_tokens)
            for v in column
            if v is not None
        )
        for column in columns
    ]
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(names)
    for row in zip(*columns, strict=True):
        writer.writerow(
            [
                int(value) if isinstance(value, bool) and not bool_columns[i] else value
                for i, value in enumerate(row)
            ]
        )
    inferred = read_csv(stream)
    series = []
    for i, (name, values) in enumerate(zip(names, columns, strict=True)):
        present = [v for v in values if v is not None]
        current = inferred[name]
        if present and all(isinstance(v, dt.datetime) for v in present):
            current = pl.Series(name, values, dtype=pl.Datetime("us"))
        elif present and all(isinstance(v, dt.time) for v in present):
            current = pl.Series(name, values, dtype=pl.Time)
        elif (
            current.dtype == pl.String and any(not isinstance(v, str) for v in present)
        ) or (current.dtype == pl.Boolean and not bool_columns[i]):
            dtype = pl.String if all(isinstance(v, str) for v in present) else pl.Object
            current = pl.Series(name, values, dtype=dtype)
        elif current.dtype == pl.Boolean and current.null_count():
            current = current.cast(pl.Float64)
        series.append(current)
    return pl.DataFrame(series), headers


def write_excel_tables(tables: dict[str, pl.DataFrame], path: Path) -> None:
    """Write ordered worksheets with the established cell and formula semantics."""
    try:
        from xlsxwriter import Workbook
    except ImportError as error:
        raise ImportError(
            "Excel writing requires xlsxwriter; install voiage[excel]."
        ) from error
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(f"Invalid extension for Excel workbook: {path.suffix}")
    with Workbook(path, {"strings_to_urls": False}) as workbook:
        header = {"bold": False}
        for name, frame in tables.items():
            worksheet = workbook.add_worksheet(name)
            if not frame.height:
                if frame.width:
                    worksheet.write_row(
                        0, 0, frame.columns, workbook.add_format(header)
                    )
                continue
            safe = frame.with_columns(
                pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None)
                for column, dtype in frame.schema.items()
                if dtype.is_float()
            )
            safe.write_excel(
                workbook=workbook,
                worksheet=worksheet,
                table_style={"style": None},
                autofilter=False,
                header_format=header,
                dtype_formats={
                    pl.Float64: "General",
                    pl.Float32: "General",
                    pl.Int64: "General",
                    pl.UInt64: "General",
                },
            )
            for row_index, row in enumerate(frame.iter_rows(), start=1):
                for column_index, value in enumerate(row):
                    if isinstance(value, float) and math.isinf(value):
                        worksheet.write_string(row_index, column_index, str(value))
                    elif isinstance(value, str) and value.startswith("="):
                        worksheet.write_formula(
                            row_index, column_index, value, None, ""
                        )
