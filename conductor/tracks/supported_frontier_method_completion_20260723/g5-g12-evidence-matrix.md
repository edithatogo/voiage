# G5–G12 evidence matrix

Audited against `origin/main` at `2638d7cf` on 2026-08-01. This matrix
separates repository-delivery evidence from the external scientific, parity,
promotion, and release gates. A green repository row does not authorize a
stable claim or issue closure.

| Gate | Repository-owned evidence | Current status | Boundary / next evidence |
|---|---|---|---|
| G5 conformance and pathology | Normative fixtures and focused tests are present for #556–#594; each phase records the relevant reference/property/pathology receipt. | Satisfied for delivered experimental slices | Add new tests only when a successor contract is approved; do not infer scientific approval from fixture presence. |
| G6 schemas, fixtures, provenance | Versioned v1 schemas, deterministic fixtures, provenance and language dispositions are recorded in each phase review and `specs/frontier/`. | Satisfied for delivered experimental slices | Residual #596–#600 have no accepted schema and remain planned. |
| G7 rights and external evidence | The plan and review artifacts explicitly retain scientific, practitioner, rights/privacy, parity, and release gates. | Pending external gates | Requires named reviewers, source authority where applicable, and signed review receipts. |
| G8 independent boundary review | Implementation reviews exist for merged experimental families and record no open repository findings. | Partial | Independent scientific review and boundary adjudication remain required per family. |
| G9 implementation or reviewed exclusion | Accepted experimental capabilities are implemented and marked experimental; residual families are classified as planned (see `reconciliation.md`). | Partial | #596–#600 require a maintainer/scientific decision before exclusion or implementation. |
| G10 polyglot dispositions | Python/Rust/R/Julia/Mojo dispositions are documented in the family contracts. | Partial | Native parity and installed shared-fixture evidence are not yet complete. |
| G11 docs/examples/discovery | Experimental APIs, CLI surfaces, docs, examples, and capability discovery are present for delivered families. | Satisfied for delivered experimental slices | Keep maturity labels experimental until promotion evidence exists. |
| G12 automated implementation review | Focused tests, lint/type checks, wheel/CLI checks, and hosted exact-head receipts are recorded for merged delivery PRs. | Satisfied for delivered experimental slices | Refresh receipts only when source heads change; final programme closeout remains governed by G13–G15. |

## Explicit non-claims

This matrix does not claim scientific validity, Rust/R/Julia/Mojo parity,
stable promotion, release publication, registry acceptance, or parent-issue
closure. Those are independent gates and remain pending where the phase plan
marks them pending.
