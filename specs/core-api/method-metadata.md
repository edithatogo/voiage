# Capability, Stability, and Maturity Metadata

## Purpose

This document defines the stable metadata contract used to explain what a
method can do, how mature it is, and whether it is exact, approximate, or
backend-dependent.

The goal is to keep bindings from inferring exactness or maturity from
implementation details that are not part of the written contract.

## Metadata Envelope

The published method-metadata payload is a bounded object with these stable
fields:

- `analysis_type`
- `method_family`
- `method_maturity`
- `approximation_status`
- `capability_labels`

Optional fields may include:

- `analysis_id`
- `decision_problem_id`
- `decision_context`
- `backend`
- `notes`

## Method Maturity

The `method_maturity` field is one of:

- `stable`
- `approximate`
- `experimental`
- `backend-dependent`

Use `stable` when the method is part of the published stable contract and
does not depend on non-contractual approximation behavior.

Use `approximate` when the published method family is scientifically useful
but is intentionally an approximation.

Use `experimental` when the method exists but is not yet ready for the stable
contract surface.

Use `backend-dependent` when the method's behavior is tied to a published
backend choice or execution path that must be surfaced explicitly.

## Capability Labels

The `capability_labels` array lists stable, machine-readable capability labels
that bindings should surface when they apply. Labels are free-form strings, but
they must remain concise and stable enough for downstream contract tests.

Examples include:

- `jax-acceleration`
- `surrogate-regression`
- `population-scaling`
- `backend-fallback`

Bindings must not hide a capability dependency by omitting the label. If a
method depends on a capability that matters to interpretation, it belongs in
`capability_labels`.

## Approximation Status

The `approximation_status` field is required and must be explicit.

Stable values are:

- `exact`
- `approximate`
- `surrogate`
- `backend-dependent`

The rule is simple: approximate methods must never look exact by omission.
If the method is approximate by design, or if the published result depends on
an approximate backend path, the payload must state that status explicitly.

## Normative Rules

1. A method-metadata payload must always include `approximation_status`.
2. Approximate or backend-dependent methods must not report `exact` as their
   approximation status.
3. Capability labels must be surfaced explicitly when they affect the meaning
   of the result.
4. Maturity and approximation are separate axes. A method may be stable and
   exact, stable and approximate, or experimental and exact.
5. This document defines the metadata contract; diagnostics remain separate and
   continue to live in `diagnostics.md`.

## Relationship To Other Contracts

Use this document together with `numerical-equivalence.md` and `diagnostics.md`:

- `numerical-equivalence.md` defines comparison rules and reproducibility
  expectations.
- `diagnostics.md` defines warnings and degraded-path reporting.
- `method-metadata.md` defines the stable capability, stability, and maturity
  metadata that bindings should surface alongside the method result.

