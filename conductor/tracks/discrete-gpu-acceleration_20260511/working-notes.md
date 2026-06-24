# Working Notes: Discrete GPU Acceleration

## Evidence Gate Status

The Discrete GPU stage remains **feasibility-only** for production acceleration
pending an Apple path signal and larger workload speedup evidence.
The Apple integrated GPU phase for this repo is currently:

- contract-preserving, with a committed CPU baseline.
- `apple_metal_vs_cpu` comparison packets in `device_comparison_available` only when
  Apple Silicon hardware is available.
- **no** confirmed Apple-side throughput gains yet from available automation runs.

The Colab accelerator validation run has now produced a hardware-backed T4 GPU
contract packet under:

- `conductor/tracks/hpc-acceleration-abstraction-contract_20260511/handoff/colab_gpu_accelerator_evidence.json`

That packet records `jax_devices == ["cuda:0"]`, `jax_platforms == ["gpu"]`,
and `cpu_evpi == jax_evpi == 1.25`. It proves JAX GPU visibility and numerical
parity for the compact EVPI validation workload. Because the integrated stage
has not produced a clear, reproducible, repeatable `apple_metal` speedup on a
production-like workload, and the Colab packet is not a production-scale
throughput benchmark, the discrete stage stays closed as a feasibility hold.

## Workload and Backend Candidates

Given existing VOI kernels, the only realistic candidates for a future discrete
GPU stage are:

- dense EVPI/ENBS reduction kernels that can map to matrix operations,
- deterministic memory-throughput scans with stable shapes,
- large, regular frontier sweeps where launch amortization can dominate overhead.

The Colab T4 evidence shows that the shared JAX path can see a GPU and preserve
the EVPI result envelope. No evidence yet shows production-sized workloads
should move this track from analysis to implementation.

## Contract Rule

CPU behavior remains the authoritative reference. Any future discrete implementation
must preserve the current result envelope exactly and remain behind the shared
abstraction contract in `hpc_acceleration_abstraction_contract.rst`.

## Transition Decision

Current decision: **Do not start production discrete-GPU implementation**.
Continue to collect reproducible `apple_metal` comparison packets and larger
JAX GPU workload packets before selecting a discrete backend.

Next-step owner: keep this track as feasibility-only and route any follow-up to
`tpu-acceleration-feasibility_20260511` or `asic-acceleration-feasibility_20260511`
only if evidence changes.
