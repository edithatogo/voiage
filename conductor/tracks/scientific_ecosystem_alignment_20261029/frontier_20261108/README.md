# Rust-authoritative VOI and comparative research programme

Date: 2026-11-08. Source baseline: `a5d87a736fa9f0f876e44872d946b2170d2a0898`.
Status: incorporated planning proposal; implementation and scientific acceptance remain open.

## Purpose and scope

Turn the maintainer's requested frontier assessment into bounded, testable delivery.
First establish a scientifically credible, Rust-authoritative VOI engine; then
investigate methods that could advance the field. Neither a broad catalogue nor
a recent dependency establishes superiority.

This is a non-HPC follow-on within the existing scientific-ecosystem track and
issue #1221, not a replacement programme or a second registry of runtime capabilities.
It refines the existing V01–V12/F01–F08 work packages in the
[integrated roadmap](../../../../docs/reviews/roadmaps/voi-methods-expansion.md).
Historical completion receipts, external gates and the 5 November HPC deferral
remain unchanged. The existing repository integrator retains ownership.

- [Ordered execution packets](plan.md)
- [Verification and dataset matrix](verification-matrix.md)
- [Research, ecosystem and publication delivery](research-publication.md)
- [Primary sources and inspection boundaries](sources.md)
- [Machine-readable packet dependencies](packets.json)

Q09 has two dependency tiers encoded in `packets.json`: `preparation_depends_on`
(Q01, Q03) covers early protocol preparation and initial software-paper work through
the existing venue track. Its `depends_on` list (Q01, Q03 and Q04–Q08) governs
completion of the full comparative programme. Preparation does not close Q09;
each publication may claim only the qualification evidence accepted for its scope.
These fields describe planning requirements, not an implemented admission engine.

## Two separate targets

**Frontier quality:** independently justified EVPI/EVPPI/EVSI and research-design
results, competitive time and memory at matched accuracy, explicit applicability,
and installed consumers using the same numerical authority. Comparison must use
current, pinned alternatives such as `voi`, BCEA and dampack where their scope
matches; unsupported comparisons are reported, not scored as losses.

**Frontier advancement:** a specific, independently reviewed methodological or
computational improvement with suitable baselines, held-out problems and a
reproducible result. Three gated research questions follow the core work:
decision-aware computation/stopping; joint research–implementation–capacity
optimisation; information value under transport, bias and competing perspectives.
Existing multilevel, quasi-Monte Carlo, Bayesian design or implementation-adjusted
methods are prior art, not new discoveries by virtue of implementation here.

## Rust-core acceptance contract

For every method advertised as native, a Rust-only installed consumer must run
its calculation, numerical policy, diagnostics and result contract without Python.
Python/R/Julia adapters must call the same authoritative kernel or carry an explicit
non-native/unsupported disposition. A common fixture does not prove common source.

Rust owns VOI estimands, validated inputs, weights and units, numerical reductions,
estimator identity, RNG/reduction policy, bounded job state, checkpoints and results.
Language front ends may own plots, notebooks, dataframe conveniences and optional
external model/posterior callbacks. External providers must declare their dependency
and scientific-semantic boundaries. Rust-to-Rust integration promises no stable ABI.

Reconcile the main numerical crate and the R package's separately bundled Rust
sources before consolidation. Preserve offline R installation through a minimal
shared crate or deterministically vendored shared source, with identity checks.
Do not replace two implementations with generated copies that can silently diverge.

Keep Arrow/Parquet as public interchange; profile typed buffers at hot boundaries
instead of assuming `.tolist()` or an Arrow label is zero-copy. Preserve versioned
API/ABI commitments and explicit migration warnings. No runtime or dependency
changes are made by this planning update.

## Architectural choices

Chosen: incremental consolidation with one shared numerical authority, a small
reference corpus and thin adapters. This preserves supported users and yields
independently reviewable changes. A wholesale rewrite adds unnecessary numerical
and compatibility risk. Continuing independent per-language formula implementations
fails the Rust-authority requirement. Adding more backends before accuracy evidence
would not address the principal scientific gap.

Follow the existing [repository ownership rules](../../agent_safe_engineering_20260907/repository-boundaries.md).
Do not import simulation, generic inference, data harvesting, surrogate training or
publication systems into the VOI core. Reuse maintained libraries after a recorded
Rust alternative, feature, MSRV, licence and numerical assessment. No automatic
Python-support widening, dependency upgrade, new repository or Hub publication is
authorised by this document.

## Acceptance and claim boundaries

A packet progresses through planned, admitted, implemented, numerically verified,
installed-consumer verified, scientifically accepted and released evidence. These
are separate claims; the plan does not change the canonical runtime maturity labels.
A missing witness remains open. Agent review is not independent external adoption
or peer review. Software correctness is not proof of clinical or policy validity.

The first delivery is Q01 capability/ownership reconciliation, alongside Q03 source
and rights preflight. Numerical implementation waits for the small shared contract.
Use at most two implementation workers and one integration pilot concurrently;
shared contracts, schemas and lockfiles have one integrator. Work is sequenced by
acceptance evidence, not speculative agent-day estimates.

## Change note

This addendum records Rust consolidation, comparative verification, candidate data,
publication routes and cross-library ownership. It does not implement an estimator,
execute a benchmark, qualify a backend, submit a paper or publish a dataset.
