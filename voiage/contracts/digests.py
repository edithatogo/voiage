"""Canonical digests for numerical contract inputs."""

from __future__ import annotations

import hashlib

import numpy as np

from voiage.contracts.critical_invariants import canonical_array_digest_input


def array_digest(values: np.ndarray) -> str:
    """Hash array dtype, shape, and canonical contiguous bytes."""
    contiguous = np.ascontiguousarray(values)
    identity = canonical_array_digest_input(
        dtype=contiguous.dtype.str,
        shape=contiguous.shape,
        payload=contiguous.tobytes(order="C"),
    )
    return hashlib.sha256(identity).hexdigest()
