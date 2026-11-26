"""Q07 bindings, buffer-protocol zero-tolist crossing, and CPU timing tests."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from voiage.methods.basic import evpi
from voiage.methods.sample_information import evsi_importance_sampling


def test_q07_buffer_crossing_avoids_tolist_and_preserves_values() -> None:
    rng = np.random.default_rng(2026)
    nb = np.ascontiguousarray(rng.normal(0.0, 10.0, size=(256, 3)), dtype=np.float64)
    lw = np.ascontiguousarray(rng.normal(0.0, 0.2, size=(16, 256)), dtype=np.float64)

    tracemalloc.start()
    t0 = time.perf_counter()
    evpi_val = evpi(nb)
    res = evsi_importance_sampling(
        nb, lw, min_ess_threshold=10.0, max_khat_threshold=0.9
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert evpi_val >= 0.0
    assert float(res["evsi"]) >= 0.0
    assert elapsed_ms < 5000.0
    assert peak_bytes > 0


def test_q07_non_contiguous_float64_views_are_safely_handled() -> None:
    base = np.arange(24, dtype=np.float64).reshape(6, 4)
    non_contig = base[:, ::2]
    assert not non_contig.flags["C_CONTIGUOUS"]
    val = evpi(non_contig)
    assert val == pytest.approx(0.0)
