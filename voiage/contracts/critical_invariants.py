"""Small production-used invariants protected by the strict mutation lane."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection


def capability_gaps(
    *,
    method_family: str,
    dtype: str,
    device: str | None,
    required_features: Collection[str],
    supported_methods: Collection[str],
    supported_dtypes: Collection[str],
    supported_devices: Collection[str],
    supported_features: Collection[str],
) -> tuple[str, ...]:
    """Return complete, deterministic reasons a backend cannot be selected."""
    missing: list[str] = []
    if method_family not in supported_methods:
        missing.append(f"method:{method_family}")
    if dtype not in supported_dtypes:
        missing.append(f"dtype:{dtype}")
    if not supported_devices:
        missing.append("device:unavailable")
    if device is not None and device not in supported_devices:
        missing.append(f"device:{device}")
    missing.extend(
        f"capability:{feature}"
        for feature in sorted(set(required_features) - set(supported_features))
    )
    return tuple(missing)


def canonical_array_digest_input(
    *, dtype: str, shape: tuple[int, ...], payload: bytes
) -> bytes:
    """Bind array dtype, shape, and canonical bytes into one digest input."""
    identity = json.dumps(
        {"dtype": dtype, "shape": shape},
        separators=(",", ":"),
    ).encode()
    return identity + b"\0" + payload


__all__ = ["canonical_array_digest_input", "capability_gaps"]
