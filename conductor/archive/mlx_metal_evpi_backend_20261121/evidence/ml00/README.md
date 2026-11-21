# ML00 decision for Apple Silicon EVPI

**No-go on 21 November 2026 AEST.** Both fixed-point prototypes met the
accuracy requirements on the workloads they admitted. Both missed the
preregistered p95 limit on the common-offset workload, which falls back to
Rust. MPS also exceeded the incremental resident-memory budget. ML04–ML06
are inapplicable for this decision. The measurements qualify the tested
workloads and device only.

The [protocol](./protocol.json) and its two drivers were committed at
`0c9678bbe37d926c766a9faa01f211827ff73447` at 14:00:26 AEST.
The declared timing run started at 14:00:51 and finished at 14:03:11.
The [raw results](./results.json) include all 21 warm samples per lane,
fresh-process timings, routes, input hashes and numerical witnesses.
Their SHA-256 is `20551f17c86df1bac3e30b2e98d91cf6151d672d6cd46d94d7f8bcf1189c31c5`.

## Device and toolchain qualification

The host was an arm64 Apple M1 Max with 24 GPU cores and 32 GiB unified memory,
running macOS 27.2.1 (26B131) and Metal 4. The run used battery power throughout.
macOS reported no thermal or performance warning. One-minute load averages
were 13.63 at the start and 8.88 at the end. This was a loaded single-host
session. Neither the cold observations nor the tail times establish a
population distribution.

The direct Rust driver and the installed Python wheel use the source at
`6a04f51c997369865665af85a11f2d732fb1f18a`. The wheel was built locally
with Maturin 1.16.0 and the repository's locked release configuration.
The direct release build linked the repository's `voiage-numerics` and
`voiage-domain` crates. Both used Rust 1.100.0
(`0ef1bc386059596dada8fb65b50425aab2ea637a`).
The direct driver's SHA-256 is
`b337517804f8a11d898ab316a186bb9511587c71ba253058e171433e614aac0b`.
The repository Cargo lock SHA-256 is
`56da7007ac4c241accd4f3a5c6f8109459c308bdcd305239b001b67e6d7fd15b`.
The direct-driver lock, derived offline from it for the two crates and the
probe package, hashes to
`d4cf29a6141ee426aa55f6d4b4a378e8d5b2e41f38ba054f3cb5b69c5de8afb0`.

The isolated arm64 CPython 3.15.2 environments reused the repository
qualification environment's read-only scientific dependencies, including
NumPy 2.5.5, Polars 2.1.0 and PyArrow 26.0.1. The cold CPU environment had
neither accelerator. The cold MLX environment contained the MLX pair, and
the MPS comparison used PyTorch with CPU operator fallback disabled.

| Installed wheel | SHA-256 | License evidence |
| --- | --- | --- |
| `mlx-0.32.4-cp315-cp315-macosx_26_0_arm64.whl` | `869c4e2d99ef622cfb39e6581afe5d3286e0c01161e03ad580561db28bcd3992` | MIT and one retained license file |
| `mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl` | `514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936` | MIT and two retained license files |
| `torch-2.15.0-cp315-cp315-macosx_14_0_arm64.whl` | `06a686eb717cd3d1662363082820be45497318887c596d691af0af0c7649c8f0` | Wheel declares Apache-2.0, Apache-2.0 with LLVM exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, ISC, MIT and Zlib. It contains 65 license-named files. |
| `voiage-3.1.0-cp315-abi3-macosx_14_0_arm64.whl` | `382fd66d4c430c609d6a643a1a96a2aed2251738aa4d5019a8ec809f659e3d81` | Apache-2.0 |

The downloaded MLX, Metal and PyTorch archives matched the PyPI digests.
PyPI exposed provenance from the `ml-explore/mlx` `release.yml` workflow
for each MLX wheel. The PyTorch wheel's provenance endpoint returned HTTP
404. The PyTorch comparison also used filelock 4.2.12, fsspec 2026.9.0,
Jinja2 3.1.6, MarkupSafe 3.0.4, mpmath 1.3.0, NetworkX 3.7, SymPy 1.14.0 and
typing-extensions 4.16.0. This was an experiment environment. A distribution
license and dependency review would be required for a proposed extra.
[MLX release files](https://pypi.org/project/mlx/0.32.4/#files) ·
[Metal release files](https://pypi.org/project/mlx-metal/0.32.4/#files) ·
[PyTorch release files](https://pypi.org/project/torch/2.15.0/#files)

Both engines returned exact signed `int64` maxima, column sums, absolute
sums and row-maximum sums in an explicit device probe using values above
`2^57`. PyTorch reported the probe outputs on MPS as `int64`. The
existing float32 `AppleMetalBackend` missed the `1e-10` relative
accuracy on the three-row analytic EVPI case and rejected the reversed
view with negative strides. Apple documents the JAX Metal plug-in as
experimental with no float64 support. It was not a device candidate in
this protocol. [Apple JAX Metal](https://developer.apple.com/metal/jax/)

## Accuracy bound and witnesses

The candidate scales finite host values by a power of two and rounds them
to integers. It admits integer reductions only when the sum of absolute
values in any reduction is bounded by `2^61`. This prevents signed overflow
for any reduction order. Each quantized value differs from the original by
at most half a quantization unit. Maxima change by at most the largest
input error, so the difference of means changes by at most one unit.
The certificate adds the Rust sequential float64 accumulation bound and
allowances for underflow and final rounding. It uses the exact GPU
magnitude reductions to bound the absolute sums and requires the combined
error to fit the G05 tolerance before returning a GPU result. The CPU
fallback handles uncertified inputs.

The [G05 register](../../../../tracks/agent_safe_engineering_20260907/numerical-case-register.json)
and exact rational witnesses covered ordinary, tied, near-tied, very small,
extreme and affine-shifted cases. Both candidate reductions matched every
admitted witness and the independent opportunity-loss reference on all
five admitted performance fixtures. Empty and single-strategy inputs
returned zero. Invalid dimensions and nonfinite values retained the same
exception classes. The scaling check matched the independent discounted
annuity calculation.

With a common offset of `1e9`, the independent three-row value remained
`2/3`. Rust returned `0.6666666269302368`, outside the G05 tolerance.
Naive float32 MLX returned zero and float64 host centering followed by
float32 MLX returned `0.6666669845581055`. Both certified routes rejected
that input. The performance fixture named common-offset used bounded
integers with a `2^20` offset. Rust returned the independent answer on
that fixture, so the fallback timing compared identical values.

## Benchmark

Times are warm medians in milliseconds. A GPU kernel time starts with
the prepared device array and includes synchronization and scalar return.
The full workflow also includes Python validation and normalization,
eligibility, conversion, upload and any CPU fallback. The raw record lists
setup, upload, first call, process elapsed and resident memory separately.

| Fixture | Rust kernel | MLX kernel | MPS kernel | Python on Rust | Python with MLX route | Python with MPS route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal 1,024 × 4 | 0.065 | — | — | 0.336 | 0.332 | 0.333 |
| Rust foundational 4,096 × 8 | 0.887 | — | — | 2.585 | 2.533 | 2.565 |
| Normal 65,536 × 4 | 6.073 | 0.461 | 1.374 | 23.889 | 3.295 | 4.834 |
| Normal 262,144 × 4 | 27.035 | 0.645 | 3.972 | 95.776 | 9.836 | 11.262 |
| Normal 65,536 × 12 | 20.247 | 0.383 | 1.611 | 55.856 | 8.468 | 10.101 |
| Fixed minor units 65,536 × 4 | 6.396 | 0.376 | 1.153 | 22.205 | 3.032 | 4.597 |
| Strided normal 65,536 × 4 | 6.203 | 0.315 | 1.550 | 22.221 | 3.154 | 4.857 |
| Common offset 65,536 × 4 | 6.231 | — | — | 23.772 | 24.166 | 25.122 |

| Preregistered gate | MLX | MPS |
| --- | ---: | ---: |
| Eligible workflows ≥ 50% | 5/8 (62.5%) | 5/8 (62.5%) |
| Equal-weight mixed median ≥ 1.5× faster | 4.65× (55.58 ms vs 258.42 ms) | 3.89× (66.51 ms vs 258.42 ms) |
| Common-offset p95 ≤ 47.83 ms | **74.24 ms** | **64.16 ms** |
| Maximum extra process peak RSS ≤ 128 MiB | 2.30 MiB | **310.19 MiB** |
| Device allocation observation | 20.01 MiB allocator peak | 40.39 MiB retained driver allocation |

All other fixtures met the warm p95 limit. For MPS, the frozen JSON
`gates.device` flag compares the retained driver allocation with the
budget. Peak device memory remains unverified. MPS independently failed
the p95 and process RSS gates. The [follow-up process profile](./memory-profile.md)
measured the PyTorch import and first-reduction contributions.

The first sampled GPU-eligible shapes occur at 65,536 rows. No crossover within the
untested gap from 4,096 to 65,536 rows was inferred. The cold numbers include
imports, allocator and driver initialization, and are descriptive.
[MLX peak memory API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html) ·
[PyTorch MPS driver memory API](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html)

The tails on the rejected common-offset case decide the result under the
frozen protocol. The extra validation and admission pass is included in
those timings. The run does not attribute the p95 difference to a single
cause. The decision retains the preregistered gate without a timing retry
or post-measurement change to workload weights.

## Reproduction

Use the attached [Python driver](./ml00.py), [Rust driver](./rust_evpi.rs)
and [protocol](./protocol.json) at the freeze commit. Build and install the
source wheel with `maturin build --release --locked --offline` and the
selected CPython 3.15.2 interpreter. Build the release Rust driver in a
separate Cargo package with path dependencies on the two crates above.
Seed its lock from the repository lock, resolve offline once, then use
`cargo build --release --locked --offline`. Place the binary at
`RUN/rust_evpi` and create `RUN/inputs`.

Use `RUN/.cpu-venv` for the CPU-only wheel and dependencies,
`RUN/.mlx-venv` for the same wheel and the MLX pair, and `RUN/.venv`
for the comparison environment with both accelerators. Retain
`mlx-device-probe.json` and `mps-device-probe.json` from the explicit
device integer probe. For the MPS probe, the driver reads its
`int64_exact` flag. The [results](./results.json) record the signed
integer sentinel outputs used in this study.

From the isolated run directory, use one native worker per thread pool
and keep PyTorch MPS operator fallback disabled.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
.venv/bin/python -I -B /path/to/ml00.py "$PWD" "$PWD/results.json"
```
