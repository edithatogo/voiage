# Contract Extension and Evolution Rules

This document defines how the stable core API can grow without breaking existing
bindings.

## Extension Principles

- Additive changes are the default: new methods, fields, and namespaces must be
  introduced in a backwards-compatible way.
- Stable v1 schemas must continue to validate existing payloads unchanged.
- New capability families should live in new, explicitly versioned schema and
  fixture subtrees.
- Optional fields are preferred over required fields when extending a stable
  contract.
- Namespaces that are intended only for experimental work must remain isolated
  from the stable v1 schema set.

## Versioning Rules

- A breaking schema change requires a new major contract version.
- New minor-scope information may be added only when older consumers can ignore
  it safely.
- A stable contract version may not silently change the meaning of an existing
  field.
- Any material change in estimator semantics must be represented as a new
  contract artifact rather than an in-place rewrite.

## Deprecation Rules

The normative timing and cross-surface requirements are defined in
`../v1/compatibility-policy.json`. A stable API must remain available for at
least two minor releases and six months, whichever is longer, and may be
removed only in the next major release.

- Deprecated fields or methods must continue to work until the replacement path
  is available in stable form.
- At v1, Python callers receive `FutureWarning`; every surface also emits a
  stable diagnostic code and supplies migration documentation. The current
  0.x EVPPI alias retains `DeprecationWarning` until the compatibility bridge
  migrates in Phase 6 of the active programme.
- Deprecation should be signaled through the diagnostics contract rather than
  by overloading successful result payloads.
- Approximate and backend-dependent methods must continue to report their
  status explicitly under the method-metadata contract.
- Removed functionality must be accompanied by a replacement or a documented
  rationale for permanent retirement.

## Stable Schema Cross-Checks

The evolution rules are written to preserve the stable v1 contract surface
defined in:

- `schemas/v1/value-array.schema.json`
- `schemas/v1/diagnostics.schema.json`
- `schemas/v1/method-metadata.schema.json`
- `schemas/v1/results/evpi.schema.json`
- `schemas/v1/results/evppi.schema.json`
- `schemas/v1/results/evsi.schema.json`
- `schemas/v1/results/enbs.schema.json`
- `schemas/v1/results/ceac.schema.json`

Any new extension contract must remain compatible with these stable schemas or
introduce a clearly versioned successor.

## Relationship To Later Fixtures

Later fixture work should treat this document as the source of truth for how
new fields and methods evolve. Deterministic fixtures may add coverage for new
capabilities, but they may not redefine the stable contract rules.
