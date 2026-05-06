Rust Parallelism And CPU Performance
====================================

This page documents the concurrency policy for the Rust core work and the
expected measurement approach for scalar, Rayon-based, and SIMD-oriented
implementations.

The goal is not to force every kernel into parallel execution. The goal is to
use parallelism where it is stable, measurable, and easier to reason about than
manual thread management.

Concurrency policy
-------------------

The Rust core should treat parallelism as an implementation detail behind the
core contract, not as part of the public result shape.

Preferred order of implementation:

1. Scalar reference implementation.
2. Deterministic batch parallelism with Rayon when the workload naturally
   decomposes across samples, strategies, thresholds, or scenarios.
3. SIMD acceleration when the inner loop is dense, uniform, and safe to vectorize.

The policy is intentionally conservative:

- Prefer a single-threaded reference path first.
- Add Rayon only where the work can be partitioned without changing result
  semantics.
- Add SIMD only when the code is numerically stable under the same tolerances as
  the scalar path.
- Do not expose thread counts, worker pools, or backend choices in the stable
  result envelope unless they affect interpretation.

Safety constraints
-------------------

Parallel code must preserve the same observable contract as the scalar path.

The following constraints apply:

- No data races or mutation of shared result state without synchronization.
- No dependence on iteration order for stable outputs.
- No assumption that Rayon will always be available at runtime.
- No vectorization shortcut that changes rounding behavior beyond the published
  tolerance envelope.
- No parallel path that silently changes diagnostics, maturity metadata, or
  reporting fields.

If a workload cannot be parallelized without changing the contract, keep the
scalar path and document the limitation.

Workloads that benefit from Rayon
---------------------------------

Rayon is a good fit for outer-loop work that is already batch-oriented:

- Monte Carlo samples that can be evaluated independently.
- Parameter draws for EVPPI-style summary work.
- Scenario batches and threshold grids.
- Independent subgroup or perspective comparisons.
- Cross-validation or bootstrap-style repetitions.

Rayon is usually not a good fit for:

- Small matrices with only a few rows or strategies.
- Kernels dominated by tiny inner loops where scheduling overhead outweighs the
  saved compute.
- Workloads that need strict ordered side effects.
- Logic that is already memory-bound and does not scale with extra threads.

SIMD measurement guidance
-------------------------

SIMD should be measured separately from Rayon so the two effects do not get
confused.

Recommended comparison set:

- Scalar reference
- Scalar plus Rayon
- Scalar plus SIMD
- Scalar plus Rayon plus SIMD, if the architecture supports it

Each variant should be benchmarked against the same deterministic inputs and
the same tolerances. Report:

- wall-clock time
- allocation count or peak memory where practical
- throughput per sample or per strategy, when that metric is meaningful
- any observed numerical differences relative to the scalar baseline

Do not declare a SIMD variant faster unless it also preserves the published
numerical contract. A small speedup that requires a looser tolerance envelope is
not a stable default.

Benchmarking expectations
-------------------------

Benchmarks should answer three questions:

1. Does the parallel version preserve the same result shape?
2. Does it remain within the contract tolerance envelope?
3. Does it provide a repeatable speedup on representative workloads?

Use representative workloads, not only synthetic microbenchmarks:

- a small case that exercises overhead
- a medium case that reflects typical use
- a larger case that shows whether parallelism scales

For each benchmark, document:

- input size
- scalar baseline result
- Rayon result
- SIMD result
- target architecture and compiler flags

When Rayon or SIMD is not worthwhile for a method family, the benchmark should
say so explicitly and preserve the scalar implementation as the reference
path.

Summary
-------

The Rust core should keep scalar behavior as the source of truth, add Rayon
only for clearly batchable workloads, and treat SIMD as a measured
optimization rather than a default architecture decision. Parallel variants
must preserve the same public contract, diagnostics, and reporting payloads as
the scalar path.

