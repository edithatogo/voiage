# G9 capability dispositions and migration guidance

## Decision boundary

G9 closes the repository-owned delivery decision for every kernel in
`specs/rust/migration_matrix.json`. It does not promote fixture-backed work to
stable parity, and it does not convert registry, signing, maintainer, or
installed-runtime gates into repository evidence.

## Dispositions

| Kernel family | Repository disposition | Evidence / current boundary | Migration guidance |
|---|---|---|---|
| `evpi` | Delivered Rust kernel; Python façade retained | Rust and Python are contract-aligned; parity is verified in the matrix | Keep the Rust scalar contract authoritative and add R/Julia installed receipts through the shared fixture packet before promotion. |
| `evppi` | Delivered Rust stable linear kernel; Python orchestration retained | Custom model, subsampling and chunking remain explicitly Python-owned | Port only after estimator, sampling and chunking semantics are frozen; do not advertise those modes as polyglot parity. |
| `evsi` | Delivered Rust aggregation kernels; Python simulation/orchestration retained | Seeded bootstrap, efficient-linear, moment and regression aggregation are Rust-backed; adaptive, random-forest and NMA paths remain Python-owned | Add a method-specific contract and fixtures before any further Rust migration or binding claim. |
| `enbs` | Delivered Rust kernel; Python façade retained | Verified contract and bridge coverage | Reuse the shared fixture and diagnostic schema for installed R/Julia runs. |
| `ceaf` | Delivered Rust kernel; Python façade retained | Verified bridge and façade coverage | Preserve result naming and validation across bindings; registry evidence remains external. |
| `dominance` | Delivered Rust family; Python façade retained | Verified bridge and façade coverage | Extend only with approved effect/cost fixtures; no new binding claim without installed evidence. |
| `value_of_heterogeneity` | Delivered Rust aggregation; Python orchestration retained | Fixture-backed parity; benchmark not yet gated | Add benchmark evidence before performance or stable-promotion claims. |
| `structural_voi` | Rust aggregation delivered; Python evaluators/scaling retained | Fixture-backed, not fully polyglot | Freeze evaluator and population-scaling contracts before porting or promotion. |
| `nma_voi` | Reviewed exclusion from Rust migration | Python-only by explicit migration matrix disposition | Keep the Python implementation supported; open a new contract-first migration task if a Rust port becomes justified. |
| `threshold_voi` | Reviewed exclusion from Rust migration | Low performance value; Python-native support retained | No port planned; revisit only with a measured workload and approved contract. |
| `validation_voi` | Reviewed exclusion from Rust migration | Low performance value; Python-native support retained | Keep Python as the supported surface and document the non-parity boundary. |
| `distributional_voi` | Reviewed exclusion from Rust migration | Python-only and no stable Rust contract | Require a distribution/tolerance contract before reconsideration. |
| `perspective_voi` | Reviewed exclusion from Rust migration | Python-only and no stable Rust contract | Preserve Python semantics; migrate only under a separately approved estimand contract. |
| `preference_voi` | Reviewed exclusion from Rust migration | Python-only and no stable Rust contract | Preserve Python semantics; migrate only with approved preference and utility fixtures. |
| Mojo binding | External boundary, no repository delivery | No local Mojo executable or binding is claimed by the v1 binding matrix | Reassess only when the upstream Mojo toolchain and Rust interop contract are available; keep capability discovery fail-closed. |

## Acceptance result

All accepted repository-owned capabilities are either delivered at their
declared maturity or have an explicit reviewed exclusion and migration path.
No unsupported method is silently represented as complete. R, Julia, Rust
PyO3, scientific promotion, signing, registry publication and parent-issue
closure remain separate follow-on gates recorded by G8/G10 and the release
closure plan.
