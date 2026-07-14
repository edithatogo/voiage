# Performance Optimization Guide

This guide provides recommendations for optimizing performance when using voiage for Value of Information analysis.

## Overview

voiage is designed for performance, but there are several strategies you can use to optimize your analyses further, especially when working with large datasets or complex models.

## Data Structure Optimization

### Use Appropriate Data Types

Ensure you're using the most efficient data types for your analysis:

```python
import numpy as np
from voiage.config import DEFAULT_DTYPE

# Use the default data type for consistency and performance
data = np.array(your_data, dtype=DEFAULT_DTYPE)
```

### Efficient Data Loading

When loading large datasets, consider using chunked loading:

```python
import xarray as xr

# Load data in chunks to manage memory usage
dataset = xr.open_dataset('large_dataset.nc', chunks={'n_samples': 1000})
```

## Computational Optimization

### Sample Size Management

For large PSA datasets, consider using a subset for initial analysis:

```python
from voiage.analysis import DecisionAnalysis

# Use a subset of samples for initial exploration
subset_indices = np.random.choice(len(all_samples), 1000, replace=False)
subset_data = all_samples[subset_indices]

analysis = DecisionAnalysis(nb_array=subset_data)
```

### Regression Sample Control

For EVPPI calculations, control the number of samples used for regression:

```python
# Use fewer samples for regression to speed up computation
evppi_result = analysis.evppi(n_regression_samples=500)
```

## Backend Optimization

### NumPy Backend

The default NumPy backend is optimized for most use cases:

```python
# Ensure you're using the NumPy backend (default)
analysis = DecisionAnalysis(nb_array=data, backend="numpy")
```

### JAX Backend

voiage supports JAX for performance optimization, which can provide significant speedups for large datasets:

```python
# Use the JAX backend for improved performance
analysis = DecisionAnalysis(nb_array=data, backend="jax")

# Enable JIT compilation for even better performance
analysis_jit = DecisionAnalysis(nb_array=data, backend="jax", use_jit=True)

# Compare performance
import time

# NumPy backend
start = time.time()
evpi_numpy = analysis_numpy.evpi()
time_numpy = time.time() - start

# JAX backend
start = time.time()
evpi_jax = analysis_jax.evpi()
time_jax = time.time() - start

# JAX backend with JIT
start = time.time()
evpi_jax_jit = analysis_jit.evpi()
time_jax_jit = time.time() - start

print(f"NumPy: {time_numpy:.4f}s")
print(f"JAX: {time_jax:.4f}s")
print(f"JAX + JIT: {time_jax_jit:.4f}s")
```

#### Performance Benefits of JAX

The JAX backend provides several performance benefits:

1. **Just-In-Time (JIT) Compilation**: Functions are compiled to optimized machine code
2. **Vectorization**: Operations are automatically vectorized for better performance
3. **GPU/TPU Support**: Can leverage accelerators when available
4. **Automatic Differentiation**: Enables gradient-based optimizations

### CPU Cluster / Distributed Execution

For workloads that need to span multiple CPU workers, use the distributed
large-scale analysis entrypoint and keep the local CPU contract unchanged:

```python
from voiage.factory import create_distributed_large_scale_analysis

analysis, cluster_config = create_distributed_large_scale_analysis(
    nb_array=data,
    chunk_size=10000,
    n_nodes=4,
    workers_per_node=8,
    scheduler="dask",
    scheduler_address="tcp://scheduler.example:8786",
)
```

The returned cluster config is an execution plan, not a new mathematical
result model. It records the selected scheduler, optional scheduler address,
and worker topology alongside the analysis object. It is suitable for cluster
schedulers, process pools, or other distributed CPU backends that preserve
ordered aggregation.

The CLI exposes the same choice via ``--scheduler process``, ``--scheduler
thread``, ``--scheduler dask``, or ``--scheduler ray`` when the selected
backend is available. ``fpga`` and ``asic`` are also recognized names in the
adapter layer, but they currently map to explicit placeholders that raise
``NotImplementedError`` until a real runtime is added.

To target a remote Dask scheduler:

```bash
voiage create-distributed-large-scale net_benefits.csv \
  --format json \
  --scheduler dask \
  --scheduler-address tcp://scheduler.example:8786
```

The JSON payload includes the same scheduler metadata under
``cluster_config`` so command output and factory return values stay aligned.

Ray uses the same scheduling slot:

```bash
voiage create-distributed-large-scale net_benefits.csv \
  --format json \
  --scheduler ray
```

FPGA and ASIC are explicit adapter names in the execution layer as well, but
they currently raise deterministic ``NotImplementedError`` failures until a
real runtime is added.

On GitHub Actions or similar CI environments, the CPU-first paths, distributed
CPU scheduler abstractions, docs, and tests are the supported targets. FPGA and
ASIC remain external-hardware tasks and should be treated as deferred follow-up
work rather than CI-backed execution modes.

#### When to Use JAX

Consider using the JAX backend when:

- Working with large datasets (>10,000 samples)
- Performing repeated calculations
- You have access to GPU/TPU hardware
- You need maximum computational performance

#### Accelerator Status for This Branch

For this code line, hardware acceleration beyond CPU is in **evidence-gated
feasibility hold**:

- Apple Metal: contract-preserving benchmark path is implemented and reviewable from
  CPU-only artifacts when hardware is unavailable.
- Discrete GPU: implemented through the shared GPU helper layer with optional
  JAX, CuPy, or PyTorch CUDA backends when those runtimes are installed.
- TPU: implemented through the existing JAX-oriented acceleration path when
  TPU devices are available; comparison evidence remains open.
- FPGA: separate execution lane; still benchmark-driven and optional.
- ASIC: contract-gated custom-circuit lane; still benchmark-driven and optional.

The repository now includes a pre-silicon evidence harness under
`hardware/pre_silicon/`. It contains a deterministic fixed-point EVPI-style RTL
kernel, CPU fixture cases, and a manifest generator for FPGA/ASIC evidence
packets. These artifacts are intended for free CI runners and Docker-based EDA
flows:

- FPGA first pass: GitHub Actions with OSS CAD Suite, Verilator, Yosys, and
  nextpnr.
- ASIC first pass: GitHub Actions with Docker, OpenROAD/OpenLane, and SKY130
  RTL-to-GDS planning.
- Fallbacks: GitHub Codespaces and Google Cloud Shell for manual debugging.

These are pre-silicon artifacts only. They do not prove physical FPGA board
runtime, fabricated ASIC runtime, or production accelerator speedup.

The roadmap decision packets live at:

- `conductor/archive/tpu-implementation_20260511/working-notes.md`
- `conductor/archive/fpga-implementation_20260511/working-notes.md`
- `conductor/archive/asic-implementation_20260511/working-notes.md`

#### Apple Metal Backend

On macOS Apple Silicon, you can also opt into the Apple Metal prototype when
PyTorch is installed with MPS support:

```python
analysis = DecisionAnalysis(nb_array=data, backend="apple_metal")
```

That backend is intentionally internal and contract-preserving; it should be
used only when you want to measure the Apple device-backed path against the
scalar baseline.

The prototype also includes a memory/throughput benchmark helper that mirrors
the committed cold/warm sample shape used by the Rust baseline artifacts.

#### Installation Requirements

To use the JAX backend, you need to install JAX:

```bash
pip install jax jaxlib
```

## Parallel Processing

### Using Multiple Cores

For operations that support it, use multiple cores:

```python
import multiprocessing as mp

# Example of parallel processing for multiple analyses
def run_analysis(data_chunk):
    analysis = DecisionAnalysis(nb_array=data_chunk)
    return analysis.evpi()

# Split data into chunks
chunks = np.array_split(large_dataset, mp.cpu_count())

# Process in parallel
with mp.Pool() as pool:
    results = pool.map(run_analysis, chunks)
```

## Memory Management

### Efficient Memory Usage

Monitor and manage memory usage for large analyses:

```python
import psutil
import gc

# Check memory usage
memory_usage = psutil.virtual_memory().percent
print(f"Memory usage: {memory_usage}%")

# Force garbage collection if needed
if memory_usage > 80:
    gc.collect()
```

## Profiling and Benchmarking

### Performance Profiling

Use Python's profiling tools to identify bottlenecks:

```python
import cProfile

# Profile your analysis
cProfile.run('analysis.evpi()', 'evpi_profile.stats')
```

### Benchmarking

Compare performance of different approaches:

```python
import time

# Benchmark different approaches
start_time = time.time()
result1 = analysis.evpi()
time1 = time.time() - start_time

start_time = time.time()
result2 = analysis.evpi(n_regression_samples=500)
time2 = time.time() - start_time

print(f"Full sample EVPI: {time1:.2f}s")
print(f"Subsampled EVPI: {time2:.2f}s")
```

### Phase 3 Handoff Benchmark Reporting

The Apple Metal track uses a unified review packet so Phase 3 can be reviewed without
Apple Silicon:

- a CPU reference contract check that any environment can run
- an optional device-backed comparison when Apple Silicon is available

Use `benchmark_mps_vs_cpu` and `compile_phase_3_handoff_packet` from
`voiage/main_backends.py` for each workload:

- `benchmark_mps_vs_cpu`
- `benchmark_memory_throughput` (swappable benchmark hook for memory packets)
- `compile_phase_3_handoff_packet` (scalar + memory handoff bundle)

#### Handoff command

Run this from repository root:

```bash
python - <<'PY'
import json
from pathlib import Path

import numpy as np

from voiage.main_backends import compile_phase_3_handoff_packet

artifact_dir = Path("bindings/rust/benches")
scalar = json.loads((artifact_dir / "scalar_cpu_baseline.json").read_text())
memory = json.loads((artifact_dir / "memory_throughput_baseline.json").read_text())

assert scalar["expected"]["evpi"] == 3.0
assert memory["expected"]["evpi"] == 3.0

nb = np.array([[10.0, 1.0], [2.0, 8.0]])
payload = compile_phase_3_handoff_packet(
    nb,
    repeats=1000,
    warmup_runs=1,
    as_json=False,
)
payload = {
    "phase": "phase_3",
    "run_context": "cpu_reference_only",
    "payload_version": "1.0.0",
    "review_phase": "phase_3",
    "review_context": "apple_metal_vs_cpu",
    "run_count": 1,
    "review": payload["review"],
    "runtime": payload["runtime"],
    "workload": payload["workload"],
    "comparison": payload["comparison"],
    "apple_metal_error": payload["apple_metal_error"],
    "scalar": payload["benchmarks"]["scalar"],
    "memory": payload["benchmarks"]["memory"],
}

print(
    json.dumps(payload, indent=2, sort_keys=True),
)
PY
```

Minimal review packet shape (fields shown):

```json
{
  "payload_version": "1.0.0",
  "review_phase": "phase_3",
  "review_context": "apple_metal_vs_cpu",
  "review": {
    "phase": "phase_3",
    "status": "cpu_reference_only",
    "required_fields": ["..."]
  },
  "workload": {"shape": [2, 2], "sha256": "...", "dtype": "float64"},
  "runtime": {"platform": "darwin", "system": "Darwin", "backend": {"torch": null}},
  "comparison": {"enabled": false},
  "apple_metal_error": {"scalar": null, "memory": null},
  "benchmarks": {
    "scalar": {
      "benchmark": "benchmark_evpi",
      "payload_version": "1.0.0",
      "workflow": "apple_metal_vs_cpu",
      "repeats": 1000,
      "warmup_runs": 1,
      "cpu": {"backend": "NumpyBackend", "repeats": 1000, "warmup_runs": 1, "result": 3.0},
      "apple_metal": null,
      "comparison": {"enabled": false}
    },
    "memory": {
      "benchmark": "benchmark_memory_throughput",
      "payload_version": "1.0.0",
      "workflow": "apple_metal_vs_cpu",
      "repeats": 1000,
      "warmup_runs": 1,
      "cpu": {
        "backend": "NumpyBackend",
        "samples": [{"phase": "cold", "iteration": 0}],
        "summary": {"evpi": 3.0}
      },
      "apple_metal": null,
      "comparison": {"enabled": false}
    }
  }
}
```

For handoff artifacts, keep each workload under
`payload["benchmarks"]["scalar"]` and `payload["benchmarks"]["memory"]` so each
workload is reproducible and comparable.

Expected key fields:

- `payload["payload_version"]`
- `payload["review_phase"] == "phase_3"` and `payload["review_context"] == "apple_metal_vs_cpu"`
- `payload["review"]["phase"] == "phase_3"` and `payload["review"]["status"]`
- `payload["runtime"]["platform"]`, `payload["runtime"]["system"]`,
  `payload["runtime"]["backend"]["torch"]`,
  `payload["runtime"]["backend"]["apple_metal_capability"]`
- `payload["workload"]["shape"]`, `payload["workload"]["dtype"]`,
  `payload["workload"]["size"]`, `payload["workload"]["sha256"]`
- `payload["benchmarks"]["scalar"]["review"]["required_fields"]` and
  `payload["benchmarks"]["memory"]["review"]["required_fields"]`
- `payload["benchmarks"]["scalar"]["cpu"]["backend"]`,
  `payload["benchmarks"]["scalar"]["cpu"]["device"]`,
  `payload["benchmarks"]["scalar"]["cpu"]["repeats"]`,
  `payload["benchmarks"]["scalar"]["cpu"]["warmup_runs"]`
- `payload["benchmarks"]["memory"]["cpu"]["backend"]`,
  `payload["benchmarks"]["memory"]["cpu"]["device"]`,
  `payload["benchmarks"]["memory"]["cpu"]["repeats"]`,
  `payload["benchmarks"]["memory"]["cpu"]["warmup_runs"]`
- `payload["benchmarks"]["scalar"]["comparison"]` and
  `payload["benchmarks"]["memory"]["comparison"]`
- `payload["benchmarks"]["scalar"]["apple_metal"]` and
  `payload["benchmarks"]["memory"]["apple_metal"]`

The track uses one unified review packet per workload (`payload["benchmarks"]["scalar"]`
and `payload["benchmarks"]["memory"]`) and a compact handoff wrapper in `payload`
for Phase 3.

Reviewer checklist (minimum accepted without hardware):

- Run the handoff command and keep output in the review packet.
- Confirm `payload["benchmarks"]["scalar"]["cpu"]["result"] == 3.0`.
- Confirm `payload["benchmarks"]["memory"]["cpu"]["summary"]["evpi"] == 3.0`.
- Verify `payload["benchmarks"]["scalar"]["cpu"]["repeats"]`,
  `payload["benchmarks"]["scalar"]["cpu"]["warmup_runs"]`,
  and `payload["workload"]["sha256"]` are set and deterministic.
- Confirm `payload["review"]["status"] == "cpu_reference_only"` if Apple hardware is
  unavailable.

#### Device-backed comparison (when available)

When Apple hardware is available, use the same command. The helper returns both the
scalar and memory comparison packets in one unified object. Compare:

- `payload["benchmarks"]["scalar"]["cpu"]["result"]` versus
  `payload["benchmarks"]["scalar"]["apple_metal"]["result"]` when available
- `payload["benchmarks"]["memory"]["cpu"]["summary"]["mean_latency_ns"]` versus
  `payload["benchmarks"]["memory"]["apple_metal"]["summary"]["mean_latency_ns"]`
- `payload["benchmarks"]["scalar"]["comparison"]["mean_latency_speedup"]`
- `payload["benchmarks"]["memory"]["comparison"]["throughput_speedup"]`

## Best Practices

1. **Start Small**: Begin with smaller datasets to test your approach
2. **Profile Regularly**: Regularly profile your code to identify performance issues
3. **Optimize Iteratively**: Make incremental improvements based on profiling results
4. **Use Appropriate Hardware**: Ensure you're using appropriate hardware for your analysis needs
5. **Cache Results**: Cache expensive computations when possible
6. **Leverage JAX**: For large-scale analyses, consider using the JAX backend with JIT compilation
7. **Monitor Memory**: Keep an eye on memory usage, especially with large datasets
