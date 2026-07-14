# Spec-to-Implementation Audit: Python Cleanup Against Spec

## Scope

This audit compares the Python package surface against the stable v1 core contract and the fixture catalog under `specs/core-api/`.

The contract scope is the stable v1 core surface:

- Schemas: `intervention`, `decision-problem`, `trial-design`, `parameter-set`, `value-array`
- Results: `evpi`, `evppi`, `evsi`, `enbs`, `ceac`

The cleanup is now recorded as complete for the stable v1 surface. The xarray
dataset boundary, optional JAX backend behavior, and fixture-format alignment
have all been verified against the current Python implementation.

## Findings

### Must-fix for v1 compliance

No blocking mismatches found in the audited Python surface.

The public API, core result plumbing, and plotting helpers inspected so far are consistent with the written v1 contract at a level that does not require an immediate behavior break.

### Compatibility alias or deprecation path

- `voiage.methods.basic.evppi(...)` accepts raw `dict[str, np.ndarray]` parameter samples in addition to `np.ndarray` and `ParameterSet`.
- That path is intentional compatibility behavior, now paired with a deprecation warning and a migration note so callers can migrate deliberately.
- If the project later wants to narrow the public surface further, that should be handled as an explicit deprecation path rather than a silent removal.

### Deferred follow-up outside the stable v1 scope

- `voiage.methods.structural` exposes structural VOI helpers that depend on model-specific evaluators and PSA samples. These are higher-level capabilities, not part of the stable v1 core contract.
- `voiage.methods.network_meta_analysis` is an advanced modeling layer and should stay outside the immediate cleanup target unless the contract expands.
- `voiage.health_economics`, `voiage.hta_integration`, and `voiage.multi_domain` are broader application layers. They are useful, but they are not core-contract blockers.
- `voiage.plot` helpers are aligned with the current VOI outputs, but they are presentation-layer utilities rather than contract-defining surface.
- Backend shims such as `voiage.backwards`/`voiage.backends.base` style compatibility layers should be treated as follow-up maintenance unless they directly block v1 compliance.

## Classification Summary

- `must-fix`: none identified
- `compatibility alias or deprecation path`: `evppi(...)` accepting raw parameter-sample dicts
- `deferred follow-up`: structural methods, network meta-analysis, HTA/multi-domain layers, plotting utilities, and backend shims

## Recommendation

Proceed to the next phase with the current public contract unchanged. If later work wants to simplify the API surface, do it behind explicit deprecations and keep the stable v1 contract as the default path.
