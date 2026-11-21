# CPU EVPI overhead on Apple Silicon

The 262,144-draw public EVPI call ran **2.79× as fast** at the median
after the first call. Peak process RSS fell by 244.34 MiB. The optimized
call deferred PyTorch loading, skipped cache hashing when caching was
disabled, and passed the contiguous float64 matrix to the existing Rust
kernel through a typed buffer. The adapter copies the data into an owned
Rust matrix and preserves the existing conversion for other layouts and
types. [Python buffer protocol](https://docs.python.org/3/c-api/buffer.html)

| Fixture and shape | Median cold call (ms) | Median warm call (ms) | Median peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| Normal, 1,024 × 4 | 3,226.07 → 1,409.23 | 0.33 → 0.13 | 492.75 → 300.55 |
| Fixed minor units, 65,536 × 4 | 3,109.27 → 1,343.07 | 23.59 → 8.40 | 515.55 → 309.66 |
| Common offset, 65,536 × 4 | 3,038.23 → 1,377.16 | 23.62 → 8.37 | 515.00 → 308.88 |
| Normal, 262,144 × 4 | 3,167.87 → 1,398.17 | 98.58 → 35.29 | 601.78 → 357.44 |

The baseline profile on the largest case attributed 17.3 ms per call
to `ndarray.tolist()` and 3.3 ms to cache hashing. The paired public calls
returned identical scalars and matched the independent ML00 reference.
PyTorch was installed across both lanes. The optimized CPU calls left it
unloaded, and explicit MPS EVPI and ENBS calls returned identical values
before and after. [Raw paired measurements](./evpi-cpu-overhead-20261121.json)

## Measurement conditions

The comparison ran on an Apple M1 Max with 32 GiB RAM and macOS 27.2.1 on
AC power on 21 November 2026. One-minute host load increased from 7.58
to 12.75. Both processes used arm64 Python 3.15.2, NumPy 2.5.5, Polars
2.1.0, PyArrow 26.0.1, JAX 0.12.1 and PyTorch 2.15.0. Native thread
pools were limited to one worker and MPS operator fallback was disabled.

The baseline Python source came from `e0fea6893a8298609b8abdb37b82733a393634d8`.
It used the retained release extension whose Rust inputs and lock match
that revision. The candidate used the same Rust 1.100.0 release profile and
macOS 14 deployment target. The raw receipt binds the native artifacts
and the four candidate source files by SHA-256.

Inputs were generated from four registered ML00 fixtures before timing.
Cold time includes the package import and first public `voiage.evpi()`
result. Peak RSS was sampled after that result. Five fresh-process pairs
per fixture alternated baseline and candidate order. Warm results come
from 31 calls in each of three fresh processes per fixture and lane
after the first call and two warmups. The original ML00 study still
governs its accelerator decision.
[ML00 protocol](../../conductor/archive/mlx_metal_evpi_backend_20261121/evidence/ml00/protocol.json) ·
[ML00 decision](../../conductor/archive/mlx_metal_evpi_backend_20261121/evidence/ml00/README.md)
