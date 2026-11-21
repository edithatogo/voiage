# ML00 follow-up memory profile

The profile reproduced the MPS process-memory gap on the M1 Max. Importing
`voiage.analysis` raised median peak RSS from 294.11 MiB without PyTorch
to 489.62 MiB with PyTorch installed on the small fixture. Both EVPI calculations
used Rust. The frozen analysis layer imports the backend module,
which imports PyTorch when installed. Importing PyTorch directly in a
separate fresh process added about 191 MiB of peak RSS.
[Frozen backend source](https://github.com/edithatogo/voiage/blob/6a04f51c997369865665af85a11f2d732fb1f18a/voiage/main_backends.py#L798)

On the admitted fixed-minor-units fixture, the full MPS route reached
630.95 MiB of peak RSS. The same-environment Rust control reached 515.00 MiB
and the CPU-only install reached 321.75 MiB. The first synchronized MPS
reduction raised the median process high-water mark from 497.72 MiB after
upload to 630.89 MiB. Physical footprint at those checkpoints rose from
402.41 MiB to 418.20 MiB. These process observations leave peak device
memory unverified.

| Fixture and environment | Operation | Peak RSS (MiB) | Physical footprint (MiB) |
| --- | --- | ---: | ---: |
| Small, CPU-only | Analysis import | 294.11 | 204.56 |
| Small, PyTorch installed | Analysis import | 489.62 | 361.56 |
| Small, PyTorch installed | Rust EVPI | 491.42 | 362.06 |
| Small, PyTorch installed | MPS route falling back to Rust | 491.89 | 362.56 |
| Fixed minor units, CPU-only | Rust EVPI | 321.75 | 220.89 |
| Fixed minor units, PyTorch installed | Rust EVPI | 515.00 | 375.70 |
| Fixed minor units, PyTorch installed | MPS EVPI | 630.95 | 418.20 |
| Common offset, PyTorch installed | MPS route falling back to Rust | 515.48 | 376.24 |

The 309.20 MiB MPS difference against the CPU-only install is consistent
with the 310.19 MiB difference in ML00. The 115.95 MiB difference against
the same-environment Rust control illustrates how much the installation
comparison includes. The frozen gate compares with the CPU-only install.
Both candidates independently exceeded the fallback p95 limit in ML00.
[Original decision](./README.md)

## Method

The follow-up ran from 18:05:23 to 18:06:37 AEST on 21 November 2026 with
the Mac on AC power. It reused the retained Voiage 3.1.0 wheel built from
`6a04f51c997369865665af85a11f2d732fb1f18a`, macOS 27.2.1 and arm64
CPython 3.15.2. NumPy 2.5.5, Polars 2.1.0 and PyArrow 26.0.1 were shared
across the environments. The comparison environment had PyTorch 2.15.0.
The retained wheel hashes matched the original qualification receipt.

Each combination ran in three fresh processes. The profiler sampled
`ru_maxrss` and macOS `footprint` at explicit checkpoints. All EVPI calls
used the frozen driver and matched the retained fixture value and route.
Imports and snapshots were deliberately separated for attribution. The
original timing samples remain the basis for the ML00 decision.
[Profiling driver](./profile_memory.py) · [All 24 process samples](./memory-profile.jsonl)

Set `EVIDENCE` to this directory and `RUN` to the isolated run directory
from the [reproduction instructions](./README.md#reproduction). For
example, the admitted MPS fixture uses this command.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
"$RUN/.venv/bin/python" -I -B "$EVIDENCE/profile_memory.py" fixed-minor-units mps
```

The `environment`, `fixture` and `backend` fields in the raw samples
identify the other invocations. The raw sample SHA-256 is
`fa704b441011d3e960414f0f7bfa7f648c4ed905ac8b163e5d1c4af16bfbc214`.
