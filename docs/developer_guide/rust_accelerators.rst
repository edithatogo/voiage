Rust Accelerators and Feasibility Boundaries
============================================

This note records the workloads that are plausibly worth accelerating in the
Rust core, and the cases where GPU, TPU, or custom-circuit work is not worth
the complexity.

The rule is simple: accelerate only the workloads that are compute-bound,
shape-stable, and already covered by a deterministic contract. Anything that
changes the contract shape or depends on irregular control flow should stay
on the scalar path.

What is plausibly GPU-bound
---------------------------

The Rust core has a few workload families that can plausibly benefit from GPU
execution if the implementation is reworked around large batches and dense
array operations:

* scalar summary kernels over many repeated PSA samples
* large matrix reductions for EVPI-style workloads
* frontier-style sweeps where the input surface is already dense and regular
* repeated dominance or CEAF scans over wide strategy sets
* batched memory/throughput profiling workloads that are dominated by the
  same arithmetic on every sample

These are the cases where the workload is mostly arithmetic, the inputs are
already numeric arrays, and the output can remain a stable summary envelope.
That makes them the only sensible candidates for accelerator feasibility
work in the short term.

Apple Metal adapter strategy
----------------------------

For the first accelerator track, the preferred strategy is to treat Apple
Metal as an internal execution adapter rather than a new public API surface.
That means:

* the Apple Metal path should target the committed scalar EVPI and
  memory/throughput baselines first;
* the CPU fallback remains authoritative and should continue to produce the
  reference result envelope;
* the Apple Metal implementation should not require public API changes just
  to route work onto the device;
* any new backend code should translate the existing Rust contract into a
  backend-specific execution plan, then return the same summary envelope.

Discrete GPU deployment assumptions
-----------------------------------

For discrete GPU execution, any future production implementation lane should
build on the existing GPU helper layer rather than introduce a brand-new public
contract.
That means:

* use the current GPU backend detection and transfer helpers as the adapter
  boundary;
* prefer JAX when it is available and the workload is dense enough to benefit
  from vectorization or compilation;
* fall back to CuPy or PyTorch CUDA where those runtimes are present;
* keep CPU fallback authoritative and preserve the same result envelope;
* document the backend choice alongside benchmark evidence so the deployment
  assumption stays reviewable.

The current Colab evidence packet validates the narrow JAX path on a T4 GPU:
``jax_devices == ["cuda:0"]``, ``jax_platforms == ["gpu"]``, and
``cpu_evpi == jax_evpi == 1.25``. This proves contract-preserving device
visibility for the compact EVPI workload. It is not yet a Rust-core production
GPU speedup claim.

Why TPU or custom-circuit support is only conditional
------------------------------------------------------

TPU support should reuse the existing JAX-oriented acceleration path when the
runtime environment exposes TPU devices. Custom-circuit support remains a
separate feasibility question. Both are only useful when the workload is
large, regular, and worth paying the control-flow and deployment cost.

That means they are conditional on all of the following:

* the workload is large enough to amortize device transfer and compilation
  overhead
* the kernel can be expressed as dense tensor math without branching over
  strategy-specific or sample-specific control flow
* the result contract can stay identical to the scalar CPU contract
* the implementation can tolerate a more rigid execution model than the CPU
  baseline

In practice, that makes TPU or custom-circuit support a follow-on feasibility
question, not a default direction. The current Rust core is still a contract-
first library, so the accelerator path must prove it can preserve the same
result envelopes and deterministic tests.

The current Colab evidence packet validates the narrow JAX path on a v5e TPU:
``jax_devices == ["TPU_0(process=0,(0,0,0,0))"]``,
``jax_platforms == ["tpu"]``, and ``cpu_evpi == jax_evpi == 1.25``. This
removes the earlier "no TPU runtime evidence" gap for the compact validation
workload, but larger workload and speedup evidence are still required before
promoting TPU as a practical acceleration lane.

Non-goals
---------

The following are explicitly not worth accelerator effort unless a later
profiling track proves otherwise:

* small scalar workloads with only a handful of strategies or samples
* code paths dominated by validation, serialization, or reporting assembly
* irregular branching logic such as dynamic study-design selection
* methods whose main cost is regression fitting or host-side orchestration
* anything that requires a new public result shape just to use the device
* host-specific optimization tricks that cannot be reproduced on Linux CI

FPGA implementation lane
------------------------

FPGA support should be treated as a separate execution lane with a narrow
kernel shape and an explicit deployment assumption. The current practical
assumptions are:

* use the same scalar result envelope as the CPU reference path;
* prefer batchable, deterministic summary kernels over irregular control flow;
* keep the backend optional and behind the shared accelerator abstraction;
* document the toolchain choice and hardware assumptions before implementation;
* require benchmark evidence before any claim of practical benefit.

The repository now exposes an explicit ``fpga`` execution-adapter name that
fails with a clear ``NotImplementedError`` until a real FPGA runtime is added.
The placeholder state is also discoverable through
``voiage.parallel.is_placeholder_execution_adapter("fpga")``.

The first repo-owned FPGA evidence path is pre-silicon only:

* committed fixed-point EVPI-style RTL and CPU fixtures under
  ``hardware/pre_silicon/``;
* Verilator lint/testbench planning;
* Yosys synthesis planning;
* nextpnr place-and-route planning; and
* GitHub Actions with OSS CAD Suite as the default free runner.

Codespaces and Google Cloud Shell are documented fallback runners for manual
debugging. Physical FPGA board runtime remains a future external evidence gate.

ASIC implementation lane
------------------------

ASIC or custom-circuit support should be treated as the most constrained
deployment lane. The implementation assumption is that it will likely reuse the
same contract-shaped summary kernels as the other accelerators, but only for
workloads that are dense, regular, and materially worth the hardware and
deployment cost.

The practical requirements are:

* preserve the CPU summary envelope as the reference contract;
* keep the backend optional and evidence-gated;
* make the deployment assumptions explicit before any implementation work;
* require reproducible CPU/ASIC comparison packets before any promotion;
* avoid any public API change just to expose a custom-circuit path.

The repository now exposes an explicit ``asic`` execution-adapter name that
fails with a clear ``NotImplementedError`` until a real ASIC/custom-circuit
runtime is added.
The placeholder state is also discoverable through
``voiage.parallel.is_placeholder_execution_adapter("asic")``.

The first repo-owned ASIC evidence path is also pre-silicon only:

* reuse of the committed fixed-point EVPI-style RTL and CPU fixtures;
* Verilator lint/testbench planning;
* Yosys synthesis planning;
* OpenROAD/OpenLane/SKY130 RTL-to-GDS planning; and
* GitHub Actions with Docker as the default free runner.

Codespaces and Google Cloud Shell are documented fallback runners for manual
debugging. Tiny Tapeout, SkyWater MPW, and fabricated-silicon runtime remain
future external evidence gates.

Those cases are either too small to benefit from offloading, or they are
shaped by control flow rather than throughput. They should remain on the CPU
until the profiling evidence says otherwise.

Practical guidance
------------------

When evaluating a Rust accelerator idea, ask these questions in order:

1. Is the workload already covered by a deterministic scalar contract?
2. Is the input shape large and regular enough to batch efficiently?
3. Can the same result envelope be produced without changing the contract?
4. Does the profiling artifact show a real gain in throughput or latency?
5. Does the memory profile still make sense after device transfer overhead?

If the answer to any of those is no, the work should stay in the scalar core
or be pushed into a separate experimental track.
