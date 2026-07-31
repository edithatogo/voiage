"""Shared safe tabular materialization helpers for built-in providers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 - public runtime annotation

import pyarrow as pa
from pyarrow import csv

from voiage.contracts.normalized_input import ResourceManifest
from voiage.ingestion.base import IngestionError, SourceAccessPolicy


def digest_file(path: Path) -> str:
    """Return a content digest without retaining sensitive source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(
    reference: str,
    policy: SourceAccessPolicy,
    *,
    sha256: str | None = None,
    byte_size: int | None = None,
    delimiter: str = ",",
    suffix: str = ".csv",
) -> pa.Table:
    """Read one declared local delimited-text resource after policy enforcement.

    Built-in providers select the delimiter and filename suffix from their
    documented profile.  This helper intentionally does not infer either from
    source bytes or a filename.
    """
    if delimiter not in {",", "\t"}:
        raise IngestionError("built-in providers support only comma or tab delimiters")
    if suffix not in {".csv", ".tsv"}:
        raise IngestionError("built-in providers support only CSV or TSV suffixes")
    if not reference.lower().endswith(suffix):
        raise IngestionError(
            f"declared resource must use the supported {suffix} filename suffix"
        )
    path = policy.materialize(reference, sha256=sha256, byte_size=byte_size)
    try:
        parse_options = (
            None if delimiter == "," else csv.ParseOptions(delimiter=delimiter)
        )
        reader = csv.open_csv(path, parse_options=parse_options)
        batches: list[pa.RecordBatch] = []
        total_rows = 0
        for batch in reader:
            total_rows += batch.num_rows
            policy.validate_tabular_batch(
                batch_rows=batch.num_rows, total_rows=total_rows
            )
            batches.append(batch)
        return pa.Table.from_batches(batches, schema=reader.schema)
    except pa.ArrowException as error:
        raise IngestionError(
            "declared delimited-text resource cannot be parsed"
        ) from error


def read_json_table(
    reference: str,
    policy: SourceAccessPolicy,
    *,
    sha256: str | None = None,
    byte_size: int | None = None,
) -> pa.Table:
    """Read the supported local JSON Table profile after policy enforcement.

    The profile is deliberately narrower than general JSON or JSON Table
    tooling: it accepts one UTF-8 JSON array whose members are object rows.
    It does not interpret JSON Pointer paths, top-level envelopes, JSON Lines,
    nested data, or descriptor-supplied parser settings.
    """
    if not reference.casefold().endswith(".json"):
        raise IngestionError("declared JSON Table resource must use a .json suffix")
    path = policy.materialize(reference, sha256=sha256, byte_size=byte_size)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IngestionError(
            "declared JSON Table resource is not valid UTF-8 JSON"
        ) from error
    if not isinstance(raw, list):
        raise IngestionError("JSON Table resource must contain a JSON array")
    if not all(isinstance(row, dict) for row in raw):
        raise IngestionError("JSON Table rows must be JSON objects")
    rows = [dict(row) for row in raw]
    if any(not all(isinstance(key, str) for key in row) for row in rows):
        raise IngestionError("JSON Table row field names must be strings")
    scalar_types = (str, int, float, bool)
    if any(
        any(
            value is not None and not isinstance(value, scalar_types)
            for value in row.values()
        )
        for row in rows
    ):
        raise IngestionError("JSON Table rows must contain only scalar values")
    policy.validate_tabular_batch(batch_rows=len(rows), total_rows=len(rows))
    try:
        return pa.Table.from_pylist(rows)
    except pa.ArrowException as error:
        raise IngestionError(
            "declared JSON Table resource cannot be converted to Arrow"
        ) from error


def materialization_receipt(
    reference: str,
    resource_id: str,
    policy: SourceAccessPolicy,
    *,
    sha256: str | None = None,
    byte_size: int | None = None,
    media_type: str = "text/csv",
) -> ResourceManifest:
    """Return immutable local-resource identity after policy resolution."""
    path = policy.materialize(reference, sha256=sha256, byte_size=byte_size)
    return ResourceManifest(
        resource_id=resource_id,
        uri=policy.source_uri(reference),
        sha256=digest_file(path),
        media_type=media_type,
        byte_size=path.stat().st_size,
    )
