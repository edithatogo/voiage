"""Deterministic pytest partition selection for parallel CI runners."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

SHARD_INDEX_ENV = "VOIAGE_TEST_SHARD_INDEX"
SHARD_COUNT_ENV = "VOIAGE_TEST_SHARD_COUNT"


def parse_shard_environment(
    environ: Mapping[str, str],
) -> tuple[int, int] | None:
    """Return a validated zero-based shard index and total shard count."""
    raw_index = environ.get(SHARD_INDEX_ENV)
    raw_count = environ.get(SHARD_COUNT_ENV)
    if raw_index is None and raw_count is None:
        return None
    if raw_index is None or raw_count is None:
        raise ValueError(
            f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} must be set together"
        )
    try:
        index = int(raw_index)
        count = int(raw_count)
    except ValueError as error:
        raise ValueError("pytest shard values must be integers") from error
    if count < 1:
        raise ValueError(f"{SHARD_COUNT_ENV} must be positive")
    if index < 0 or index >= count:
        raise ValueError(f"{SHARD_INDEX_ENV} must be in the range [0, {count})")
    return index, count


def shard_for_nodeid(nodeid: str, shard_count: int) -> int:
    """Map a pytest node ID to one stable shard independent of collection order."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count
