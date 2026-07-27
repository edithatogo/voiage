"""Shared safe tabular materialization helpers for built-in providers."""

from __future__ import annotations

import hashlib
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
) -> pa.Table:
    """Read a declared CSV file after policy enforcement."""
    if not reference.lower().endswith(".csv"):
        raise IngestionError("built-in providers currently support CSV resources only")
    path = policy.materialize(reference, sha256=sha256, byte_size=byte_size)
    try:
        return csv.read_csv(path)
    except pa.ArrowException as error:
        raise IngestionError("declared CSV resource cannot be parsed") from error


def materialization_receipt(
    reference: str,
    resource_id: str,
    policy: SourceAccessPolicy,
    *,
    sha256: str | None = None,
    byte_size: int | None = None,
) -> ResourceManifest:
    """Return immutable local-resource identity after policy resolution."""
    path = policy.materialize(reference, sha256=sha256, byte_size=byte_size)
    return ResourceManifest(
        resource_id=resource_id,
        uri=policy.source_uri(reference),
        sha256=digest_file(path),
        media_type="text/csv",
        byte_size=path.stat().st_size,
    )
