# Working Notes: Discrete GPU Acceleration

## Evidence Gate Status

The Discrete GPU stage remains **feasibility-only** pending an Apple path signal.
The Apple integrated GPU phase for this repo is currently:

- contract-preserving, with a committed CPU baseline.
- `apple_metal_vs_cpu` comparison packets in `device_comparison_available` only when
  Apple Silicon hardware is available.
- **no** confirmed Apple-side throughput gains yet from available automation runs.

Because the integrated stage has not produced a clear, reproducible, repeatable
`apple_metal` speedup on a production-like workload, the discrete stage stays
closed as a feasibility hold.

## Workload and Backend Candidates

Given existing VOI kernels, the only realistic candidates for a future discrete
GPU stage are:

- dense EVPI/ENBS reduction kernels that can map to matrix operations,
- deterministic memory-throughput scans with stable shapes,
- large, regular frontier sweeps where launch amortization can dominate overhead.

No evidence yet shows these should move this track from analysis to implementation.

## Contract Rule

CPU behavior remains the authoritative reference. Any future discrete implementation
must preserve the current result envelope exactly and remain behind the shared
abstraction contract in `hpc_acceleration_abstraction_contract.rst`.

## Transition Decision

Current decision: **Do not start implementation**. Continue to collect reproducible
`apple_metal` comparison packets before selecting a discrete backend.

Next-step owner: keep this track as feasibility-only and route any follow-up to
`tpu-acceleration-feasibility_20260511` or `asic-acceleration-feasibility_20260511`
only if evidence changes.
