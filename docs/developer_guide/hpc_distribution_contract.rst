HPC Distribution Contract
=========================

This page defines the HPC-facing contract for ``voiage``. It is the written
reference for what the project should look like when someone asks whether it is
ready for HPC-adjacent distribution, curation, or registry handoff.

The contract intentionally stays conservative:

* ``voiage`` is HPC-deployable and HPC-friendly.
* ``voiage`` is not HPC-native.
* CPU-first portability matters more than speculative accelerator work.
* GPU and TPU claims must be scoped to the evidence available. The current
  Colab packets prove JAX device visibility and EVPI parity on T4 GPU and v5e
  TPU, but they do not prove production speedup.
* No production accelerator or custom-circuit claim should be made without
  benchmark and deployment evidence that justifies a dedicated implementation
  track.
* This contract does **not** imply that live Spack/EasyBuild/HPSF/E4S submissions are already complete.

Target ecosystems
-----------------

The contract treats the following ecosystems as the relevant HPC-facing
distribution targets:

* Spack
* EasyBuild
* HPSF
* E4S

Spack and EasyBuild are the concrete packaging paths. HPSF and E4S are the
curation / visibility paths that sit on top of those package recipes.

Required contract outcomes
--------------------------

The HPC-facing version of the project must be able to state:

* the package is installable on standard HPC systems without hidden network
  fetches during build time;
* the source release is deterministic and reproducible;
* dependency versions are pinned or otherwise recorded clearly enough for site
  deployment;
* the runtime model remains CPU-first unless benchmark evidence supports a new
  accelerator track;
* cluster/distributed execution uses a scheduler-facing adapter boundary rather
  than hard-coded MPI/SLURM logic in the core API;
* unsupported accelerator schedulers such as ``fpga`` and ``asic`` are exposed
  as explicit placeholders rather than hidden runtime claims;
* the next accelerator track uses a standard backend abstraction decision and does
  not fragment into incompatible runtime-specific contracts;
* all accelerator escalation decisions are captured in the
  `hpc_acceleration_abstraction_contract.rst <hpc_acceleration_abstraction_contract.rst>`_
  reference contract;
* the package is suitable for recipe-based distribution through Spack and
  EasyBuild;
* the curation story for HPSF and E4S is explicit and does not overclaim
  registry status.
* GitHub CI is used to validate CPU-first and distributed CPU paths, Colab
  evidence validates the compact GPU/TPU JAX path, and FPGA/ASIC remain
  deferred hardware-backed follow-up work.

Acceptance criteria
-------------------

The contract is satisfied when:

* the repo can describe itself as HPC-deployable and HPC-friendly without
  claiming HPC-native status;
* the HPC distribution matrix covers Spack, EasyBuild, HPSF, and E4S;
* no track claims direct live registry/curation completion unless the external
  maintainer action is confirmed;
* accelerator escalation criteria are written down before any accelerator
  implementation is proposed;
* the release and roadmap docs point readers at this contract instead of
  implying that HPC registry status is already solved.
* Apple Metal Phase-3 handoff evidence uses the unified review packet emitted
  by ``benchmark_mps_vs_cpu``, with ``runtime`` metadata and ``review`` status
  preserved and no public API shape change. Handoff evidence should be stored
  as ``phase_3_cpu_reference.json`` and ``phase_3_handoff_bundle.json`` in the
  Apple Metal prototype track directory.
* The Apple Metal prototype track is complete. Apple Silicon MPS acceleration
  evidence remains pending until Apple Silicon capture is available for the
  integrated GPU optimization track.
* Colab GPU/TPU visibility and parity evidence is stored under
  ``conductor/tracks/hpc-acceleration-abstraction-contract_20260511/handoff/``
  and currently proves T4 GPU and v5e TPU runtime visibility plus EVPI parity;
  these Colab packets are not timing, warm-up, or throughput review packets.
* The HPC registry readiness evidence packet is maintained as external handoff
  work, not as a repository-owned publish operation.

Roadmap to HPC-native
---------------------

The contract does not claim HPC-native status today. The staged path to that
claim is documented in `hpc_native_roadmap.rst <hpc_native_roadmap.rst>`_ and
starts with Apple integrated-GPU optimization through Metal-backed execution
before moving on to discrete GPU, TPU, and ASIC feasibility work.

Out of scope
------------

* Building or maintaining a real HPC cluster deployment.
* Hard-coding a single distributed scheduler backend into the stable contract.
* Implementing production GPU, TPU, or custom-circuit kernels beyond the
  compact JAX validation path.
* Claiming HPC-native status before there is benchmark-backed evidence for it.
