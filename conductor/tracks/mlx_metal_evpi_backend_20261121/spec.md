# Specification — Optional MLX/Metal EVPI backend

## Goal

Determine whether Apple Silicon can accelerate the existing unweighted,
risk-neutral EVPI workflow at its current accuracy. Keep the stable Rust
float64 path as the execution authority. An optional experimental route is
eligible for implementation only after a go decision in ML00.

## Evidence so far

- The [5 November experiment](../../../docs/reviews/mlx-metal-evpi-scope-20261105.md)
  ran MLX 0.32.4 on an M1 Max with Python 3.15.1. The naive float32 path
  lost the EVPI signal after a common `1e9` shift. Host float64 centering
  recovered the signal, with an absolute difference of `1.19e-7` from the
  installed Rust result, outside the G05 tolerance.
- The exploratory timing compared the public Python CPU call with a GPU
  reduction. The public path materializes nested lists and the Rust kernel
  sorts values before summing. Their separate costs were unmeasured. The
  experiment established neither numerical equivalence nor a speedup.
- [`AppleMetalBackend`](../../../voiage/main_backends.py) already provides
  float32 EVPI and ENBS through PyTorch MPS. The MLX work merged in
  [PR #1277](https://github.com/edithatogo/voiage/pull/1277) covers scope and
  probes. The [existing Metal evidence contract](../../archive/apple-metal-production-speedup-evidence_20260625/index.md)
  supplies the reporting baseline.

## Numerical acceptance

Freeze the [G05 case register](../agent_safe_engineering_20260907/numerical-case-register.json),
the current Rust reference revision, and independent analytic results.
Retain the G05 relative tolerance of `1e-10` and absolute tolerance of
`1e-12`. Exercise near-zero EVPI, near ties, common offsets, shuffled rows
and strategies, extreme finite scales, input dtypes and layouts, population
scaling, and existing empty, single-strategy and invalid-input behavior.

Require an input-specific eligibility rule for GPU execution. A conservative
error bound must account for host preprocessing, quantization, device
reduction, and restoration of units. Observed test errors alone cannot
certify a new input. Use the existing Rust path whenever the bound cannot
certify accuracy, including unresolved near-zero results or out-of-range
inputs. Preserve current validation errors and test accuracy against
analytic references as well as Rust behavior. Check error before the
nonnegative clamp and verify population scaling separately. Charge the
eligibility check to the GPU workflow.

## ML00 feasibility decision

Before timing, record the target workload shapes and layouts, representative
data or deterministic generators, and the required end-to-end improvement,
eligible fraction, fallback overhead and peak-memory budget. State the number
of repetitions and how variation will be reported.

On a named Apple Silicon device, run both of these comparisons on the same
cases and at matched accuracy.

1. **Kernel timing.** Compare Rust on a prebuilt `SampleMatrix` with the GPU
   calculation on prepared input. Report setup separately. In the existing
   Rust smoke benchmark, input construction is inside the timed closure.
2. **Workflow timing.** Compare the supported Python EVPI call with the
   candidate from equivalent host input through a scalar result. Include
   validation, eligibility, copies and conversion, synchronization, return
   of the result, and fallback. Record cold and warm latencies, variability,
   peak memory, eligible fraction and crossover sizes. Count fallback calls
   in the mixed workload.

Record the source revision, device and OS, Python and dependency versions,
artifact hashes, fixtures and raw measurements. A **go** requires numerical
acceptance and the predeclared practical benefit on a representative workload.
A missing device or missing required evidence for a surviving candidate
is **blocked**. A completed study that finds no safe, worthwhile path is a
**no-go** and satisfies feasibility. If no candidate clears the numerical
gate, retain the failed witnesses and close feasibility without speed claims.
Keep the CPU path and its acceptance independent. An EVSI or other heavier
workload needs its own profiling evidence, estimand and accuracy contract.
Colab cannot supply Apple Metal device evidence.

## Conditional execution design

Explicitly opt into a single experimental EVPI route at the existing
validated per-decision boundary, before the common population scaling step.
[`DecisionAnalysis.evpi()`](../../../voiage/analysis.py) calls Rust
directly. Registering a `Backend` subclass alone cannot route that call.
Reuse the capability descriptor where it fits and keep the per-call decision
explicit. Preserve the scalar return, current errors, scaling and cache
behavior. Keep the existing MPS and ENBS callers compatible.

Make the selected device and reason for an eligible execution or Rust fallback
observable through existing diagnostics without exposing input values. Lazy-load
the optional accelerator and test clean base installs, missing devices and
packages, ineligible inputs, and genuine calculation errors. Keep weighted
EVPI, other estimands and R/Julia dispatch outside this proposal.

## Reuse maintained libraries

- Use MLX public reductions and force completion with `mx.eval`. Evaluate
  `mx.compile` when profiling identifies a likely benefit and include
  compilation cost in the timing. PyPI lists CPython 3.15 macOS arm64
  [MLX wheels](https://pypi.org/project/mlx/0.32.4/#files) and matching
  [Metal wheels](https://pypi.org/project/mlx-metal/0.32.4/#files).
  Qualify the actual wheels, licenses and device in the selected repository
  Python 3.15 runtime. Reuse NumPy and `math.fsum` for host arithmetic when
  needed and keep the existing Arrow/Polars interchange.
- Compare the existing [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html)
  path under the same accuracy rule. Prefer one supported Metal route when
  ML00 establishes a benefit. Keep the experimental
  [JAX Metal plug-in](https://developer.apple.com/metal/jax/) as a documented
  option if qualified. It does not support float64.
- Use the existing [Rust benchmark](../../../rust/crates/voiage-numerics/benches/foundational.rs),
  `pytest-benchmark`, and Metal evidence schema and validator. Defer a
  portability layer, custom kernels and new benchmarking dependencies until
  a measured need justifies them. Add a platform-gated extra only after a go.

[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
documents the forced evaluation used for timing.
