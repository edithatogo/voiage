# G5–G12 evidence matrix

Audited against `origin/main` at `366186b3` on 2026-08-01. The exact
per-family artifact map is `g5-g13-evidence-map.json`. This matrix
separates repository-delivery evidence from the external scientific, parity,
promotion, and release gates. A green repository row does not authorize a
stable claim or issue closure.

| Gate | Repository-owned evidence | Current status | Boundary / next evidence |
|---|---|---|---|
| G5 conformance and pathology | All 18 families map to focused reference/property/pathology evidence. | Repository-owned gate satisfied | Do not infer scientific approval from fixture presence. |
| G6 schemas, fixtures, provenance | All 18 families map to versioned schemas/fixtures or their dedicated portable contract. | Repository-owned gate satisfied | Successor estimands require new governed contracts. |
| G7 rights and external evidence | Family boundaries retain scientific, practitioner, rights/privacy, parity and release gates. | Repository record satisfied; external gates pending | Requires named external evidence and review receipts. |
| G8 evidence and boundary review | All 18 families map to independent reference or implementation review. #571 has a four-pass exact-commit review, #595 has independent phase reviews with no remaining Critical/High/Medium findings, and #619 has a fresh exact-commit remediation re-review. | Repository-owned gate satisfied | Scientific/design/classification review remains a separate external gate. |
| G9 implementation or reviewed exclusion | All 18 accepted families have merged experimental delivery evidence; no exclusion is invented. | Repository-owned gate satisfied | Experimental maturity remains unchanged. |
| G10 polyglot dispositions | Every family maps to a capability/binding disposition artifact. | Repository record satisfied; parity pending | Rust/R/Julia native parity and installed shared-fixture evidence remain open. |
| G11 docs/examples/discovery | Every family maps to experimental documentation or its contract README. | Repository-owned gate satisfied | Keep maturity labels experimental until promotion evidence exists. |
| G12 automated implementation review | Existing focused, repository-harness and hosted delivery receipts are mapped per family. | Repository-owned gate satisfied | Fresh programme exact-head assurance is G14 and remains pending. |

## Explicit non-claims

This matrix does not claim scientific validity, Rust/R/Julia/Mojo parity,
stable promotion, release publication, registry acceptance, or parent-issue
closure. Those are independent gates and remain pending where the phase plan
marks them pending.
