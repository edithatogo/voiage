# G12 broader-validation blocker report

## Run

`tox -e coverage_report` ran 3,889 tests with 15 skips and 94.51% total
coverage (above the 90% threshold), but exited non-zero with 21 failures.

## Findings

| Category | Failures | Disposition |
|---|---|---|
| Native estimation-variance bridge | `evsi_var` runtime, pathological-input, replay-digest and CLI tests fail because the loaded native `compute_evsi_variance` accepts a different arity than the Python façade supplies. | High-priority repository blocker. Rebuild/install the matching PyO3 extension and rerun the focused estimation suite; do not promote this surface until fixed. |
| Conductor/archive baseline | Projection, cross-reference, v1 baseline, supported-frontier and registry-audit tests reference archived track paths or stale generated projections. | Repository baseline reconciliation blocker; refresh projections/fixtures against the current archive state, preserving historical receipts. |
| Binding/readiness metadata | Julia BinaryBuilder, language dispositions and v2 export determinism tests disagree with current manifests/metadata. | Reconcile generated metadata and support claims; do not infer registry approval. |
| Perspective determinism | Fixture manifest/payload is not current with the test expectation. | Regenerate only from the canonical source packet and record hashes. |

## Options

- **Recommended:** fix the native extension/source mismatch first, then refresh
  deterministic Conductor projections and rerun the full coverage gate.
- **Alternative:** defer the estimation-variance surface and run coverage with
  an explicitly documented exclusion; this preserves safety but leaves G12
  incomplete.
- **Not recommended:** waive the failures because aggregate coverage exceeds
  90%; that would conceal runtime and governance drift.

G12 remains open for remediation. Hosted checks, scientific review, registry
approval and parent closure remain separate external gates.
