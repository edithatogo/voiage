# G12 broader-validation blocker report

## Run

`tox -e coverage_report` ran 3,889 tests with 15 skips and 94.51% total
coverage (above the 90% threshold), but exited non-zero with 21 failures.

## Findings

| Category | Failures | Disposition |
|---|---|---|
| Native estimation-variance bridge | `evsi_var` runtime, pathological-input, replay-digest and CLI tests initially failed because the loaded native `compute_evsi_variance` accepted a different arity than the Python façade supplied. | **Resolved locally** by rebuilding the matching PyO3 extension; focused estimation runtime/surface tests pass. |
| Conductor/archive baseline | Projection, cross-reference and supported-frontier tests referenced archived track paths. | **Resolved locally** by making projections and tests archive-aware and updating generated paths. The independent v1 programme baseline still lists archived tracks as active and remains open. |
| Binding/readiness metadata | Julia BinaryBuilder, language dispositions and v2 export determinism tests disagree with current manifests/metadata. | **Resolved locally** for the focused suite: generated Julia `Manifest.toml` removed, archive disposition paths reconciled, v2 exports and registry snapshot pass. |
| Perspective determinism | Fixture manifest/payload is not current with the test expectation. | **Resolved locally** by regenerating the canonical fixture payload; deterministic test passes. |

## Options

- **Recommended:** fix the native extension/source mismatch first, then refresh
  deterministic Conductor projections and rerun the full coverage gate.
- **Alternative:** defer the estimation-variance surface and run coverage with
  an explicitly documented exclusion; this preserves safety but leaves G12
  incomplete.
- **Not recommended:** waive the failures because aggregate coverage exceeds
  90%; that would conceal runtime and governance drift.

G12 remains open only for the independent v1 programme baseline snapshot, which still records archived tracks as active. Hosted checks, scientific review, registry approval and parent closure remain separate external gates.
