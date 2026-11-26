"""Q02 offline voiageR staticlib vs voiage-numerics Rust-authority parity tests."""

from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from voiage.methods.basic import evpi

ROOT = Path(__file__).resolve().parents[1]
VOIAGER_RUST_MANIFEST = ROOT / "r-package" / "voiageR" / "src" / "rust" / "Cargo.toml"


def _build_voiager_cdylib(work_dir: Path) -> Path:
    """Compile the offline no_std voiageR Rust crate into a callable shared library."""
    cargo_bin = shutil.which("cargo")
    cc_bin = shutil.which("cc")
    assert cargo_bin is not None
    assert cc_bin is not None

    target_dir = work_dir / "cargo-target"
    subprocess.run(
        [
            cargo_bin,
            "build",
            "--release",
            "--offline",
            "--manifest-path",
            str(VOIAGER_RUST_MANIFEST),
            "--target-dir",
            str(target_dir),
        ],
        check=True,
    )
    staticlib = target_dir / "release" / "libvoiage_r_native.a"
    assert staticlib.is_file()

    # Create a tiny C shim re-exporting voiage_rust_evpi and voiage_rust_enbs into a shared lib.
    shim_c = work_dir / "shim.c"
    shim_c.write_text(
        """
extern int voiage_rust_evpi(const double *values, int rows, int columns, double *out);
extern int voiage_rust_enbs(double evsi_result, double research_cost, double *out);
int q02_evpi(const double *values, int rows, int columns, double *out) {
    return voiage_rust_evpi(values, rows, columns, out);
}
int q02_enbs(double evsi_result, double research_cost, double *out) {
    return voiage_rust_enbs(evsi_result, research_cost, out);
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    shared = work_dir / "libq02_voiager.so"
    subprocess.run(
        [
            cc_bin,
            "-shared",
            "-O2",
            "-fPIC",
            str(shim_c),
            str(staticlib),
            "-o",
            str(shared),
        ],
        check=True,
    )
    return shared


def test_voiager_offline_rust_matches_voiage_numerics_witnesses(tmp_path: Path) -> None:
    lib_path = _build_voiager_cdylib(tmp_path)
    lib = ctypes.CDLL(str(lib_path))
    lib.q02_evpi.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.q02_evpi.restype = ctypes.c_int
    lib.q02_enbs.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.q02_enbs.restype = ctypes.c_int

    def call_voiager_evpi(arr: np.ndarray) -> tuple[int, float]:
        flat = np.ascontiguousarray(arr, dtype=np.float64)
        out = ctypes.c_double(0.0)
        status = lib.q02_evpi(
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            flat.shape[0],
            flat.shape[1],
            ctypes.byref(out),
        )
        return int(status), float(out.value)

    # 1. Canonical Q02 witness [[0, 100], [100, 0]] -> 50.0
    crossover = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float64)
    status, r_evpi = call_voiager_evpi(crossover)
    assert status == 0
    assert abs(r_evpi - 50.0) <= 1.0e-12
    assert abs(r_evpi - float(evpi(crossover))) <= 1.0e-12

    # 2. Ties and non-negativity
    ties = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    status, r_ties = call_voiager_evpi(ties)
    assert status == 0
    assert r_ties == 0.0

    # 3. Extreme finite matrix: must match voiage-numerics without overflowing to inf
    scale = 0.75 * float(np.finfo(np.float64).max)
    extreme = np.array([[scale, 0.0], [0.5 * scale, scale]], dtype=np.float64)
    expected = 0.25 * scale
    py_extreme = float(evpi(extreme))
    status, r_extreme = call_voiager_evpi(extreme)
    assert status == 0
    assert np.isfinite(r_extreme)
    assert abs(py_extreme - expected) / expected <= 1.0e-12
    assert abs(r_extreme - expected) / expected <= 1.0e-12

    # 4. Catastrophic cancellation parity (Python/Rust/R all match Neumaier 0.25)
    cancelling = np.array(
        [[1.0e20, 1.0e20], [1.0, 0.0], [0.0, 1.0], [-1.0e20, -1.0e20]],
        dtype=np.float64,
    )
    py_cancelling = float(evpi(cancelling))
    status, r_cancelling = call_voiager_evpi(cancelling)
    assert status == 0
    assert abs(py_cancelling - 0.25) <= 1.0e-12
    assert abs(r_cancelling - 0.25) <= 1.0e-12

    # 5. Overflow before clamp must fail explicitly in both Python/Rust and R
    max_f64 = float(np.finfo(np.float64).max)
    overflow = np.array(
        [[max_f64, 0.0], [0.0, max_f64], [0.0, max_f64]],
        dtype=np.float64,
    )
    status, _ = call_voiager_evpi(overflow)
    assert status != 0
    with pytest.raises((ValueError, RuntimeError)):
        evpi(overflow)

    # 6. Invalid NaN / Inf / ENBS checks
    status, _ = call_voiager_evpi(np.array([[np.nan, 1.0]], dtype=np.float64))
    assert status != 0

    enbs_out = ctypes.c_double(0.0)
    assert lib.q02_enbs(5000.0, 6000.0, ctypes.byref(enbs_out)) == 0
    assert abs(enbs_out.value - (-1000.0)) <= 1.0e-12
    assert lib.q02_enbs(10.0, -1.0, ctypes.byref(enbs_out)) != 0
