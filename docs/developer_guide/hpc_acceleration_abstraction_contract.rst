HPC Acceleration Abstraction Contract
====================================

Purpose
-------

This contract preserves one runtime shape for every accelerator path and prevents
incompatible hardware-specific behavior from fragmenting the public API.

Current completion decision:

* The abstraction contract track is complete.
* The current concrete progression is:
  `apple-metal-backend-prototype_20260510` (complete), then
  `apple-metal-integrated-gpu-optimization_20260511` (in progress).
* Discrete GPU, TPU, and ASIC tracks are active but explicitly follow-on and
  remain constrained by this abstraction contract.

The contract applies to:

* Apple integrated GPU (Metal on macOS),
* discrete CUDA/ROCm-class GPU paths,
* TPU via compiler-backed execution,
* and ASIC/custom-circuit feasibility.

Standard interface
-----------------

All accelerator tracks must preserve this contract:

* CPU is the reference implementation.
* A track may only provide execution acceleration when output envelopes and
  diagnostics are bit-for-bit equivalent (within documented tolerance) to the
  CPU reference for existing deterministic payloads.
* Public APIs remain unchanged by backend selection.
* Benchmark packets must include:

  * runtime metadata (`payload_version`, `runtime`, and `review.status`),
  * workload fingerprint (shape and hash),
  * comparison fields (`speedup`, `delta`, and `status`).

* On failure or unsupported paths, fallback to CPU must be automatic and explicit.

Standardized acceleration candidates
-----------------------------------

The first-generation contract decision is:

1. Keep NumPy as the required portable baseline in-core implementation.
2. Keep Apple Metal as the integrated-GPU first-class pilot path using the existing
   platform-native adapter.
3. Evaluate a standard compiler path for dense-kernel workloads before any
   broad discrete GPU or TPU implementation.

Practical consequence: the accelerator stack should converge on a common
execution abstraction first, and only then split into hardware-specific
deployment slices where justified.

The immediate "single-library-first" candidates are:

* **JAX / XLA** for dense tensor program lowering and TPU eligibility.
* **PyTorch MPS / CUDA APIs** for pragmatic incremental execution when dense matrix
  pathways are already available.
* **XLA-style IR boundaries** where supported by target hardware.

No single vendor API should define the contract by itself. The shared abstraction
is the only contract-boundary decision; backend engines may evolve underneath it.

Transition criteria to feasibility follow-on
------------------------------------------

For each hardware class, progression from feasibility to implementation requires:

1. A preserved contract comparison packet (CPU + target device) with identical
   result schema.
2. Reproducible benchmark evidence that includes warm-up and repeatable timings.
3. Clear owner + next-step fields for unresolved deployment or registration tasks.
4. A public decision record in `Conductor` describing why the class stays
   optional or proceeds.

HPC roadmap dependency
----------------------

This contract is the upstream dependency for:

* `discrete-gpu-acceleration_20260511`,
* `tpu-implementation_20260511`,
* `asic-implementation_20260511`,
* and `apple-metal-integrated-gpu-optimization_20260511`.
