"""Canonical digests for numerical contract inputs."""

# pyright: reportExplicitAny=false, reportAny=false

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from voiage.contracts.critical_invariants import canonical_array_digest_input


def array_digest(values: NDArray[np.generic]) -> str:
    """Hash array dtype, shape, and canonical contiguous bytes."""
    # Preserve np.ascontiguousarray's scalar shape without materializing a
    # complete copy of a strided array or concatenating its bytes twice.
    normalized = values.reshape(1) if values.ndim == 0 else values
    header = canonical_array_digest_input(
        dtype=normalized.dtype.str, shape=normalized.shape, payload=b""
    )
    digest = hashlib.sha256(header)
    chunks = np.nditer(
        normalized,
        flags=["external_loop", "buffered", "zerosize_ok", "refs_ok"],
        order="C",
        # Bound buffers by bytes, not elements; one oversized element is the
        # unavoidable minimum for NumPy's iterator.
        buffersize=max(1, 65_536 // max(normalized.dtype.itemsize, 1)),
    )
    for chunk in chunks:
        digest.update(cast("NDArray[np.generic]", chunk).tobytes(order="C"))
    return digest.hexdigest()
