# Six-gate single-maintainer closeout plan

This plan separates repository evidence from accountable decisions and external
venue outcomes. Subagents may prepare advisory reports, but this repository has
one maintainer and does not infer independent approval from panel participation.

## Ordered gates

- [ ] **G1 Scientific validity:** freeze the candidate; bind specifications,
  references, fixtures, assumptions, tolerances, adversarial cases, findings and
  the advisory panel packet to one commit; record the maintainer's promote,
  defer or reject decision.
- [ ] **G2 Cross-language parity:** run clean installed Rust/Python/R/Julia
  fixtures where runtimes exist; capture canonical envelopes, versions,
  tolerances, pathological cases, unavailable-runtime receipts and hashes.
- [ ] **G3 Promotion:** apply the capability/promotion matrix only after G1 and
  G2; record a separate hash-bound maintainer maturity decision.
- [ ] **G4 Release:** validate exact-tag artifacts, SBOM, provenance, checksums,
  reproducible builds, security evidence and clean-install receipts.
- [ ] **G5 Publication and registry:** submit venues one at a time and record
  only authoritative receipts for arXiv, JOSS, CRAN/R-universe, conda-forge,
  Yggdrasil, SciCrunch/RRID and related issue #555 work.
- [ ] **G6 Issue/Conductor closure:** reconcile every issue, parent/child,
  Project item, PR/head, acceptance criterion and evidence hash; close or
  archive only after the relevant gate receipt exists.

## Advisory panel protocol

Use role-separated subagents for scientific semantics, numerical assurance,
API/parity, reproducibility/security and publication metadata. The orchestrator
records disagreement, uncertainty and limitations. Reports are advisory
artifacts, not independent approvals, signatures or venue receipts.

## Decision and contingency rules

- Any mismatch, unresolved high-severity finding, missing receipt or disputed
  scientific claim keeps the gate pending/deferred.
- Local green tests, a merged PR, panel synthesis or Project status never imply
  scientific approval, stable promotion, release, publication, registry
  acceptance or issue closure.
- Keep #313–#323 and #841/#843–#849 open until their acceptance evidence exists.
- Do not silently reinterpret #325's closed state while its body contains
  unchecked acceptance items; document the inconsistency and obtain the
  maintainer's explicit reopen/retain decision.
