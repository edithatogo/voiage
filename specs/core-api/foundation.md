# Core API Foundation

## Purpose

This document is the written foundation for the cross-language `voiage` core API. It exists so later schema files, fixture sets, and bindings are derived from an explicit contract rather than from incidental behavior in the current Python implementation.

## Audience

This foundation is for:

- maintainers deciding the canonical semantics of the library
- contributors authoring JSON Schema and fixture artifacts
- binding authors working in Python, R, Julia, and later languages
- review tooling and smaller execution models that need a compact normative reference

## Relationship To Later Artifacts

- `decision-record.md` captures the fixed choices that should not be re-opened in downstream implementation tracks.
- The next schema track turns this foundation into machine-readable structures.
- The conformance-fixture track turns the same contract into executable examples and cross-language checks.

## Scope

The v1 contract is net-benefit-first and treats study-design and research-decision objects as first-class concepts.

The contract is intentionally written before implementation details are finalized so that:

- Python remains the reference binding, not the only source of truth
- schema names stay stable across runtimes
- fixture behavior is compared against the written specification, not inferred heuristics

## Canonical Vocabulary

The stable v1 contract uses a small vocabulary that later tracks must preserve:

- `study-design`: inputs describing the design of the evidence-generating process, including sample size, arm structure, and observation model assumptions
- `research-decision`: the decision problem framed in terms of net benefit, willingness-to-pay, and candidate strategies
- `parameter-set`: named uncertainty inputs that can be sampled, fixed, or grouped
- `strategy`: a decision alternative whose net benefit can be compared against the others
- `outcome`: the cost, effect, and derived net-benefit outputs for a strategy under a given analysis context

The contract is written in net-benefit-first terms. Cost and effect remain available, but they are interpreted through the decision problem rather than as the top-level public abstraction.

## Stable V1 Method Families

The following capability families are in-scope for v1 and should be treated as stable:

- EVPI
- EVPPI
- EVSI
- ENBS and sample-size optimization
- CEAC and CEAF
- CE plane and expected loss / opportunity loss views
- NMB and ICER-oriented outputs where they support the core decision workflow
- multi-intervention comparison

The following families are reserved as extension points or later-phase work:

- structural uncertainty VOI
- network-meta-analysis-driven VOI
- adaptive design VOI
- portfolio optimization
- heterogeneity / value-of-heterogeneity extensions beyond the core contract
- estimation-problem VOI where it requires a separate semantic model

This split keeps v1 credible against the current open-source leaders while avoiding premature expansion into experimental surfaces.

## Design Constraints

- The core model must stay language-agnostic.
- JSON Schema is the machine-readable contract layer.
- Markdown is the normative layer for meaning, invariants, and workflow rules.
- Python, R, and Julia are the initial runtime targets.
- Xarray-labeled arrays, NumPy, optional JAX acceleration, and Arrow/Parquet interchange are part of the foundation, but only as explicitly scoped contracts.
- Polars remains adapter-only tabular tooling and is not the canonical compute model.

## Capability Baseline

The minimum credible bar for `voiage` v1 is that it must match or exceed the documented core VOI and decision-analysis coverage of the strongest packages in the comparison set:

- `voi` for breadth of VOI methods, sample-size optimization, and backend flexibility
- `dampack` for expected-loss workflows and decision-curve style analysis
- `BCEA` for richer CEA reporting and mixed-analysis capabilities
- `TreeAge Pro` for mature commercial decision-analysis ergonomics
- `hesim` and `heemod` for broader health-economic modeling workflows and interoperability

Concretely, the contract must make the following non-negotiable:

1. EVPI, EVPPI, and EVSI are first-class, not optional add-ons.
2. Sample-size and ENBS-style optimization are part of v1 scope.
3. CEAC, CEAF, and expected-loss outputs are available alongside VOI metrics.
4. The contract supports multiple strategies and a coherent net-benefit comparison workflow.
5. The written contract must be stable enough that downstream bindings do not need to infer semantics from Python-only helpers.

The competitor feature scan is therefore an input to scope control, not a license to reproduce every experimental surface immediately.

## Downstream Outputs

The next tracks are expected to produce:

1. Canonical schemas for the v1 contract.
2. Conformance fixtures that test those schemas and the semantics in this document.
3. Python cleanup that aligns the current implementation with the written contract.

## Small-Model Execution Guide

Later tracks must not re-litigate the following choices unless a spec revision is explicitly opened:

- Net-benefit-first is the public conceptual frame.
- Markdown semantics and JSON Schema are the only authoritative contract layers.
- Python, R, and Julia are the initial runtime targets.
- Xarray-labeled arrays, NumPy, optional JAX, Arrow/Parquet, and adapter-only Polars are fixed infrastructure choices.
- v1 stable scope includes VOI metrics and CEA outputs, but not every extension family in the comparison scan.
- Any behavior not expressible in the written contract is implementation detail, not contract surface.
