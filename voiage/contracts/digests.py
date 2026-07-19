"""Canonical digests for numerical contract inputs."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def array_digest(values: np.ndarray) -> str:
    """Hash array dtype, shape, and canonical contiguous bytes."""
    contiguous = np.ascontiguousarray(values)
    identity = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": contiguous.shape},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(identity)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()
