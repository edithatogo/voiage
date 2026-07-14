# Working Notes: Apple Metal Integrated GPU Optimization

## Baseline Workloads

The first Apple Metal pass should focus on workloads that are already
deterministic, numeric, and contract-stable:

- the committed scalar EVPI baseline in `bindings/rust/benches/scalar_cpu_baseline.*`
- the committed memory/throughput baseline in
  `bindings/rust/benches/memory_throughput_baseline.*`

These are the current CPU comparison anchors for the integrated GPU work.
They already reflect the stable public envelope and can be used to judge
whether a Metal-backed path is worth expanding.

## Candidate Expansion Shapes

If the Apple path proves promising, the next plausible workload families are:

- larger EVPI-style matrix reductions
- repeatable memory/throughput scans over deterministic summary kernels
- later, frontier-style sweeps only if the workload remains dense and regular

## Contract Rule

The CPU fallback stays authoritative. Any Metal-backed path must preserve the
same contract shape and must not change the public result envelope.

## Metal Path Draft

The first Metal-backed design should be treated as an internal execution
adapter, not a new public API surface:

- target the committed scalar EVPI baseline and memory/throughput baseline
- keep the CPU fallback authoritative for every workload in scope
- avoid public API changes just to route work onto the device
- translate the existing Rust contract into a backend-specific execution plan
  and return the same summary envelope

## Apple Deployment Requirements

Any Apple-specific packaging story should assume:

- macOS Apple Silicon hosts for build and validation
- Metal-capable system libraries available at runtime
- reproducible release artifacts that still ship the same Rust contract
- CPU fallback coverage preserved in CI so Apple-only code paths stay
  contract-safe

## Benchmark Status

The phase-3 CPU-only comparison evidence is now available from:

- `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_cpu_reference.json`
- `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_handoff_bundle.json`
- `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_runtime_freeze.txt`
- `conductor/archive/apple-metal-integrated-gpu-optimization_20260511/handoff/phase_3_evidence_manifest.json`

A device-backed comparison is still pending:

- `Apple` hardware with `torch.backends.mps.is_available() == true`.
- one full run of the same handoff command with `apple_metal` payloads populated.

## No-Hardware Closure Gate

- [x] CPU-only comparison packet has been generated and pinned:
  - `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_cpu_reference.json`
  - `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_handoff_bundle.json`
  - `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_runtime_freeze.txt`
- [x] The evidence manifest explicitly marks `status: cpu_reference_only` and
  `pending` device-backed run(s).
- [x] The phase-3 review packet exists in this track at:
  `conductor/archive/apple-metal-integrated-gpu-optimization_20260511/handoff/phase_3_evidence_manifest.json`.
- [x] The track can advance to reviewer-ready "CPU-first" closure until hardware is
  available.

## Deferred Device-Backed Work (when Apple Silicon is available)

- Run both workloads once with `torch.backends.mps.is_available() == true` and
  record populated `apple_metal` fields in:
  - `conductor/archive/apple-metal-backend-prototype_20260510/handoff/phase_3_handoff_bundle.json`
  - `conductor/archive/apple-metal-integrated-gpu-optimization_20260511/handoff/phase_3_evidence_manifest.json`
- Update the corresponding manifest `status`/`pending` fields and this section with
  the achieved speedup/delta values.

## Phase 3 Handoff Pack (Reviewer-Ready)

Phase 3 is now defined as a review packet that can be validated from git history
and local CPU-only runs, so a reviewer does not need full Apple hardware to
validate readiness.

### Required review artifacts

Attach or cite these files when reviewing the phase:

- `bindings/rust/benches/scalar_cpu_baseline.json`
- `bindings/rust/benches/memory_throughput_baseline.json`
- `bindings/rust/benches/scalar_cpu_baseline.rs`
- `bindings/rust/benches/memory_throughput_baseline.rs`
- `voiage/main_backends.py` (benchmark helpers and helper payload schema)
- `tests/test_apple_metal_backend.py`

### Minimum CPU-only review command set

These commands validate the current contract without a Metal device:

```bash
# Check the committed scalar baseline contract still matches expected EVPI
python - <<'PY'
import json
from pathlib import Path
from voiage.main_backends import NumpyBackend, benchmark_evpi, benchmark_memory_throughput
import numpy as np

artifact_dir = Path("bindings/rust/benches")
scalar = json.loads((artifact_dir / "scalar_cpu_baseline.json").read_text())
memory = json.loads((artifact_dir / "memory_throughput_baseline.json").read_text())

assert scalar["expected"]["evpi"] == 3.0
assert memory["expected"]["evpi"] == 3.0
print("cpu baselines loaded:", scalar["benchmark_name"], memory["benchmark_name"])

nb = np.array([[10.0, 1.0], [2.0, 8.0]])
payload_evpi = benchmark_evpi(NumpyBackend(), nb, repeats=1000, warmup_runs=1)
payload_memory = benchmark_memory_throughput(NumpyBackend(), nb, repeats=1000, warmup_runs=1)
print("cpu evpi benchmark result:", payload_evpi["result"])
print("cpu throughput:", payload_memory["summary"]["throughput_ops_per_sec"])
print("cpu memory samples:", len(payload_memory["samples"]))
PY
```

### Expected handoff content

Every Phase-3 review packet should include:

- A CPU baseline verification (as above) showing contract anchors match artifacts.
- If Apple Silicon was available, a second run that prints/records the Metal-backed
  payload from the same benchmark helpers for direct comparison.
- A short, explicit table of what is and is not compared:

  - CPU backend result
  - device/backend reported (`NumpyBackend` vs `AppleMetalBackend`)
  - cold-start vs warm-start throughput
  - reproducibility fields (`repeats`, `warmup_runs`, `benchmark helper fields`)

If Apple hardware is unavailable, the handoff is still acceptable when it contains
the CPU contract checks above and a clear note that a device-backed code path to time or compare against the baseline is deferred.
