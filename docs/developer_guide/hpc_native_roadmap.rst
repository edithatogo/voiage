HPC Native Enablement Roadmap
=============================

This roadmap turns the current HPC-friendly posture into a staged path toward
an HPC-native claim. The repository should not treat HPC-native as a current
status. It should treat it as an evidence-backed endpoint that requires
progressive accelerator work and benchmark proof.

Current completion decision:

* `apple-metal-backend-prototype_20260510` is complete and has delivered the
  Phase-3 CPU reference artifacts for deterministic review.
* `apple-metal-integrated-gpu-optimization_20260511` has completed the CPU-first
  Phase-3 review packet and is awaiting Apple Silicon + MPS device access for a
  full ``apple_metal`` payload capture.
* Colab validation has now captured contract-preserving CPU/JAX parity on a
  T4 GPU and a v5e TPU through the shared JAX-oriented path. These artifacts
  prove device visibility and result parity for the validation workload; they
  do not yet prove production-scale speedup.
* FPGA and ASIC now expose explicit adapter placeholders, but their real
  runtimes and comparison evidence remain open.

The registry-deployment program is the prerequisite stage. Once the release
submission tracks are complete, this roadmap can proceed with accelerator work
on top of a stable deployment baseline.

Roadmap sequence
-----------------

1. Integrated GPU optimization on Apple Silicon

   * prioritize Metal-backed acceleration and Apple-friendly integrated GPU
     paths first
   * prove that the contract stays stable on the CPU fallback path
   * establish representative workloads and repeatable benchmark gains
   * start from the committed ``scalar_cpu_baseline`` and
     ``memory_throughput_baseline`` artifacts in ``bindings/rust/benches/``
     as the initial CPU comparison set

2. Discrete GPU acceleration

   * expand beyond integrated GPUs only after the Metal path is measured and
     contract-safe
   * compare backend candidates where the workload shape justifies it
   * use a shared abstraction contract for backend execution to avoid one-off
     hardware-specific implementation drift

3. TPU implementation

   * reuse the existing JAX-oriented acceleration path where TPU devices are
     available
   * require proof that device transfer and compilation overhead are worth it
   * keep CPU fallback authoritative and the public contract stable

4. FPGA implementation

   * define the toolchain and kernel shape before implementation widens
   * keep the backend optional and benchmark-driven

5. ASIC / custom-circuit implementation

   * treat as the last-stage execution lane
   * require strong evidence that the workload is regular enough to justify
     the hardware and deployment cost
   * confirm fallback contract and benchmark packet compatibility before any
     non-CPU implementation slice is approved

Current evidence status
-----------------------

As of the Colab capture on 2026-06-23 UTC, the reviewer-closure decisions are:

* **CPU reference**: the deterministic scalar CPU baselines remain the
  authoritative comparison set through the Apple Metal prototype handoff.
* **Discrete GPU**: the implementation lane is archived
  (`conductor/archive/discrete-gpu-implementation_20260511/`), and a
  hardware-backed Colab T4 validation packet now confirms JAX GPU device
  visibility, EVPI parity, and placeholder-adapter reporting:
  ``conductor/tracks/hpc-acceleration-abstraction-contract_20260511/handoff/colab_gpu_accelerator_evidence.json``.
  This is a contract validation artifact, not a production speedup claim.
* **TPU**: implementation lane through the JAX-oriented backend path with
  hardware-backed Colab v5e validation now confirming JAX TPU device
  visibility, EVPI parity, and placeholder-adapter reporting:
  ``conductor/tracks/hpc-acceleration-abstraction-contract_20260511/handoff/colab_tpu_accelerator_evidence.json``.
  This closes the "no TPU runtime evidence" gap for the compact validation
  workload, while larger workload and speedup evidence remain future work.
* **FPGA**: implementation lane with explicit adapter placeholder exposure and
  free CI pre-silicon evidence complete. The committed evidence path includes
  a fixed-point EVPI-style RTL kernel, CPU fixtures, Verilator simulation
  planning, Yosys synthesis planning, nextpnr place-route planning, and probe
  manifests (`conductor/tracks/fpga-implementation_20260511/`). Physical FPGA
  board runtime remains a future external evidence gate.
* **ASIC / custom-circuit**: implementation lane with explicit adapter
  placeholder exposure and free CI pre-silicon evidence complete. The committed
  evidence path reuses the fixed-point RTL kernel and CPU fixtures, records
  Verilator/Yosys/OpenROAD/OpenLane/SKY130 RTL-to-GDS command status targets,
  and includes probe manifests (`conductor/tracks/asic-implementation_20260511/`).
  Tiny Tapeout, SkyWater MPW, and fabricated-silicon runtime remain future
  external evidence gates.

The GPU and TPU artifacts are indexed by
``conductor/tracks/hpc-acceleration-abstraction-contract_20260511/handoff/colab_accelerator_evidence_manifest.json``.
Remaining holds are now about repeatable acceleration, production-sized
workloads, physical FPGA board runtime, and fabricated ASIC evidence, not about
basic JAX device visibility for GPU or TPU or about repo-owned FPGA/ASIC
pre-silicon scaffolding.

Active follow-through tracks
----------------------------

The completed setup and readiness tracks are no longer the place to prove live
speedup or hardware outcomes. The active production-evidence tracks are:

* ``hpc-production-speedup-evidence-program_20260625`` for the shared
  benchmark packet contract, CPU fallback proof, and production-speedup gating.
* ``cpu-cluster-production-benchmark-evidence_20260625`` for larger CPU
  cluster and distributed scheduler workloads.
* ``apple-metal-production-speedup-evidence_20260625`` for Apple Silicon
  Metal/MPS timing, throughput, warm-up, and CPU comparison packets.
* ``discrete-gpu-production-speedup-evidence_20260625`` for CUDA-class GPU
  evidence through Colab or other available runners.
* ``tpu-production-scale-colab-evidence_20260625`` for production-scale TPU
  evidence using ``colab`` and, when project/quota/billing access exists,
  ``gcloud``.
* ``accelerator-evidence-automation_20260625`` for common JSON schemas,
  artifact ingestion, ``gh run watch`` monitoring, and blocked-run reporting.
* ``fpga-physical-board-runtime-evidence_20260625`` for board runtime,
  bitstream, CPU parity, and speedup evidence when hardware exists.
* ``asic-mpw-shuttle-and-silicon-evidence_20260625`` for Tiny Tapeout or
  SkyWater MPW submission state, fabricated silicon, and runtime evidence.
* ``custom-circuit-production-acceleration-review_20260625`` for the final
  go/no-go decision on FPGA/ASIC production acceleration claims.

Until those tracks produce production-sized benchmark packets, the correct
status remains setup/visibility/parity evidence rather than HPC-native speedup.

Apple deployment requirements
-----------------------------

The Apple integrated-GPU stage should be packaged and validated with these
requirements in mind:

* macOS Apple Silicon hosts for build and validation
* Metal-capable system libraries available at runtime
* reproducible release artifacts that still ship the same Rust contract
* CPU fallback coverage preserved in CI so Apple-only code paths stay
  contract-safe

Benchmark status
----------------

The current Apple Metal track has established the committed CPU baselines and the
contract-preserving adapter strategy. Phase-3 remains the review milestone where
the Metal-backed implementation is measured against the CPU reference.

The Colab accelerator validation notebook has separately produced lightweight
JAX comparison evidence:

* T4 GPU: ``jax_devices == ["cuda:0"]``, ``jax_platforms == ["gpu"]``,
  ``cpu_evpi == jax_evpi == 1.25``.
* v5e TPU: ``jax_devices == ["TPU_0(process=0,(0,0,0,0))"]``,
  ``jax_platforms == ["tpu"]``, ``cpu_evpi == jax_evpi == 1.25``.

Those runs validate runtime visibility and numerical parity for the compact
EVPI workload. They should be used as hardware-evidence anchors for the shared
JAX path, but they are not sufficient by themselves to claim HPC-native
acceleration or production workload speedup.

Standardized accelerator stack
------------------------------

The repository is expected to use one shared backend abstraction policy for
future GPU/TPU/FPGA/ASIC decisions. See
`hpc_acceleration_abstraction_contract.rst <hpc_acceleration_abstraction_contract.rst>`_
for the shared policy. A dedicated Conductor track captures that policy
before implementation widens beyond Apple Metal, so tracks are evaluated against
one compatibility and reproducibility contract instead of ad-hoc backend-specific
logic.

The policy should explicitly name the fallback behavior for:

* dense tensor workloads,
* kernel launch and compile overhead,
* unsupported control-flow patterns,
* and deterministic benchmark packet emission.

Phase 3 comparison handoff
---------------------------

Before marking this track as accelerated, Phase-3 review should pass through the
artifact bundle below so Phase 3 can be reviewed without full Apple Silicon access:

1. Confirm baseline anchors are still loadable and unchanged:

   - ``bindings/rust/benches/scalar_cpu_baseline.json``
   - ``bindings/rust/benches/memory_throughput_baseline.json``
   - ``bindings/rust/benches/scalar_cpu_baseline.rs``
   - ``bindings/rust/benches/memory_throughput_baseline.rs``
   - ``tests/test_apple_metal_backend.py``
   - ``conductor/tracks/apple-metal-backend-prototype_20260510/handoff/phase_3_cpu_reference.json``
   - ``conductor/tracks/apple-metal-backend-prototype_20260510/handoff/phase_3_handoff_bundle.json``
   - ``conductor/tracks/apple-metal-backend-prototype_20260510/handoff/phase_3_runtime_freeze.txt``

2. Confirm the review helper contract in ``voiage/main_backends.py`` still exposes:

   - ``benchmark_evpi``
   - ``benchmark_memory_throughput``
   - ``payload_version``, ``benchmark``, ``review``, ``runtime``, and ``workflow`` fields from ``benchmark_mps_vs_cpu``
   - ``backend``/``device`` fields for device selection traceability
   - ``mean_latency_ns`` and ``throughput_ops_per_sec`` metrics
   - ``comparison`` speedup and delta fields for review packets
   - a single unified packet returned by ``benchmark_mps_vs_cpu``

Minimal reviewer packet shape:

.. code-block:: json

    {
      "backend": "apple_metal_vs_cpu",
      "payload_version": "1.0.0",
      "workload": {"shape": [2, 2], "sha256": "..."},
      "runtime": {"platform": "darwin", "backend": {"torch": "..."}},
      "review": {
        "phase": "phase_3",
        "status": "device_comparison_available|cpu_reference_only",
        "required_fields": ["backend", "..."]
      },
      "cpu": {"backend": "NumpyBackend", "result": 3.0},
      "apple_metal": null,
      "review_context": "apple_metal_vs_cpu",
      "comparison": {"enabled": false}
    }

3. Deliver a review report with:

   - the CPU baseline result check against the committed JSON artifact
   - the Metal-backed payload fields if the reviewer has device access
   - an explicit statement of any skipped steps if hardware is unavailable
   - ``workload.sha256`` and ``runtime.platform`` metadata for reproducibility

4. Confirm that no API surface change is required for benchmarking and that the
   CPU fallback remains authoritative.

The track can be promoted from Phase 3 once the handoff report is attached to
the reviewer notes and every field above has been validated against the current
artifacts.

Contract rules
--------------

* CPU behavior stays the reference path until an accelerator proves itself.
* The public result envelope must not change just because a new backend is
  introduced.
* Benchmark evidence, reproducibility, and deployment realism come before any
  HPC-native claim.
* Apple integrated GPUs are the first accelerator target because they give the
  team a lower-risk path to validate backend plumbing, memory layout, and
  contract preservation before moving to more ambitious hardware.
* The initial workload anchors are the deterministic EVPI scalar baseline and
  the committed memory/throughput baseline; both stay CPU-first until a Metal
  path proves a repeatable gain without changing the contract shape.

Where this roadmap lives
------------------------

The current HPC contract is documented in
`hpc_distribution_contract.rst <hpc_distribution_contract.rst>`_.
The implementation tracks for each stage are registered in the Conductor
tracks file and should be executed in order.

Registry-readiness track alignment
----------------------------------

The HPC registry path is coordinated separately from accelerator implementation by:

* `hpc-registry-readiness_20260511`
* `spack-registry-readiness_20260511`
* `easybuild-registry-readiness_20260511`
* `hpsf-curation-readiness_20260511`
* `e4s-curation-readiness_20260511`

These tracks are required so the project can move from "ready in principle" to
"ready for maintainer handoff" without overclaiming direct publication status.
