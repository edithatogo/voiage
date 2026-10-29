"""Byte compatibility and bounded-memory checks for calculation identities."""

import hashlib
import tracemalloc

import numpy as np
import pytest

from voiage.contracts.critical_invariants import canonical_array_digest_input
from voiage.contracts.digests import array_digest


@pytest.mark.parametrize(
    "kind", ["contiguous", "strided", "empty", "scalar", "big-endian"]
)
def test_digest_retains_existing_framing(kind: str) -> None:
    values = np.arange(60, dtype=np.float64).reshape(10, 6)
    if kind == "strided":
        values = values[::-2, ::2]
    elif kind == "empty":
        values = values[:0]
    elif kind == "scalar":
        values = np.asarray(3.0)
    elif kind == "big-endian":
        values = values.astype(">f8")
    contiguous = np.ascontiguousarray(values)
    expected = hashlib.sha256(
        canonical_array_digest_input(
            dtype=contiguous.dtype.str,
            shape=contiguous.shape,
            payload=contiguous.tobytes(order="C"),
        )
    ).hexdigest()
    assert array_digest(values) == expected


def test_digest_does_not_copy_the_entire_strided_input() -> None:
    values = np.arange(2_000_000, dtype=np.float64)[::2]
    tracemalloc.start()
    try:
        array_digest(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < values.nbytes // 2


@pytest.mark.parametrize(
    "values",
    [
        np.asfortranarray(np.arange(30, dtype=np.int64).reshape(5, 6)),
        np.arange(30, dtype=np.float32).reshape(5, 6).T,
        np.broadcast_to(np.array([1.0, 2.0]), (12, 2)),
        np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
        np.arange(6).astype("S20000")[::2],
    ],
    ids=["fortran", "transpose", "broadcast", "complex", "wide-element"],
)
def test_digest_matches_canonical_bytes_across_layouts(values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    expected = hashlib.sha256(
        canonical_array_digest_input(
            dtype=contiguous.dtype.str,
            shape=contiguous.shape,
            payload=contiguous.tobytes(order="C"),
        )
    ).hexdigest()
    assert array_digest(values) == expected


def test_digest_bounds_buffer_for_wide_elements() -> None:
    values = np.zeros(2_000, dtype="S10000")[::2]
    tracemalloc.start()
    try:
        array_digest(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


def test_repeated_digests_do_not_retain_input_sized_buffers() -> None:
    import gc

    values = np.arange(2_000_000, dtype=np.float64)[::2]
    tracemalloc.start()
    try:
        first = array_digest(values)
        gc.collect()
        initial, _ = tracemalloc.get_traced_memory()
        for _ in range(30):
            assert array_digest(values) == first
        gc.collect()
        final, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert final - initial < 1_000_000
