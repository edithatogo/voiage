# Diagnostics, Warnings, and Approximation Caveats

## Purpose

This document defines the stable diagnostics payloads used by the core API to
report unsupported capabilities, degraded execution paths, and approximation
caveats.

The diagnostic contract is separate from reproducibility metadata:

- reproducibility explains how a result was produced
- diagnostics explain whether the result is trustworthy, degraded, or only
  approximately comparable

## Diagnostic Envelope

The published diagnostics payload is a bounded object with these stable fields:

- `analysis_id`
- `status`
- `warnings`
- `unsupported_capabilities`
- `degraded_paths`
- `approximation_caveats`
- `backend` when a backend-specific execution path matters

The diagnostic envelope must not silently omit a degraded or approximate
execution path.

### Status Values

The `status` field is one of:

- `ok`
- `degraded`
- `unsupported`
- `approximate`

Use `ok` only when the reported result does not rely on fallback behavior or an
approximation caveat.

Use `degraded` when the result is usable but one or more capabilities were
reduced, substituted, or partially supported.

Use `unsupported` when the requested capability could not be executed under the
current contract.

Use `approximate` when approximation is the intended method family or the
published result is only scientifically meaningful under an explicit tolerance
or surrogate model.

## Warning Records

The `warnings` array carries stable, user-facing warning records.

Each warning record must include:

- `severity`
- `code`
- `message`

Optional warning fields may include:

- `capability`
- `degraded_path`
- `approximation`
- `backend`
- `fallback`

Stable warning severities are:

- `info`
- `warning`
- `critical`

The warning `code` should be short, machine-readable, and stable across
bindings. The code identifies the class of problem, while the message explains
the specific user-facing fallback or caveat.

## Normative Rules

1. Unsupported capabilities must be listed in `unsupported_capabilities`.
2. Fallback execution paths must be listed in `degraded_paths`.
3. Approximation caveats must be listed explicitly in
   `approximation_caveats`.
4. Approximate methods must never appear exact by omission.
5. If a result is approximate by design, the diagnostics payload must not use
   `status: "ok"`.
6. If the payload reports a degraded or approximate path, the warning list must
   include at least one record that explains the reason in plain language.

## Example Fragment

```json
{
  "analysis_id": "evsi-screening-001",
  "status": "degraded",
  "backend": "numpy",
  "warnings": [
    {
      "severity": "warning",
      "code": "backend_fallback",
      "message": "JAX is unavailable; falling back to NumPy.",
      "capability": "jax-acceleration",
      "backend": "numpy",
      "fallback": "numpy"
    },
    {
      "severity": "warning",
      "code": "approximation_caveat",
      "message": "Regression surrogate estimates are tolerance-bounded, not exact.",
      "approximation": true,
      "degraded_path": "surrogate-regression"
    }
  ],
  "unsupported_capabilities": [
    "jax-acceleration"
  ],
  "degraded_paths": [
    "surrogate-regression"
  ],
  "approximation_caveats": [
    "Result uses a published absolute tolerance of 1e-10 and relative tolerance of 1e-8."
  ]
}
```

## Relationship To Later Metadata

Capability, stability, and maturity metadata are defined in
`method-metadata.md`. This document focuses only on the stable diagnostic and
warning payloads that bindings must preserve.
