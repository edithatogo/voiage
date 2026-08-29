# G7 evidence-gate ledger

G7 records the gates that constrain Rust/polyglot delivery. Recording a gate
does not satisfy it; only the named authority or an auditable receipt can move
its state.

| Gate | Scope | Authority/owner | State | Evidence or next action |
|---|---|---|---|---|
| Source rights and licences | Rust/Python/R/Julia source, fixtures, examples and bundled metadata | Maintainer and upstream rights holders | `recorded; pending confirmation where applicable` | Retain licence/provenance manifests; confirm any newly acquired fixture rights before redistribution. |
| Privacy and data handling | Dataset fixtures, receipts, diagnostic payloads and logs | Maintainer/data custodian | `fail_closed` | Use synthetic/offline fixtures by default; do not record credentials, PII or raw remote payloads. |
| Scientific validity | Estimands, numerical tolerances, stochastic seeds and maturity promotion | Scientific/contract review panel | `pending` | Panel reviews the complete parity packet before `fixture_backed` becomes `verified` or experimental becomes stable. |
| Practitioner suitability | Decision interpretation, domain examples and unsupported-method warnings | Practitioner reviewers/maintainer | `pending` | Review examples and user-facing warnings; do not infer approval from tests or documentation build. |
| Installed runtime parity | Rust/Python/R/Julia shared fixtures and ABI/layout | Maintainer plus hosted runners | `partial` | Rust/Python/Julia local evidence exists; complete clean R package and exact shared-fixture packet, then rerun panel review. |
| Hosted CI and merge | Exact commit, protected checks, resolved threads and merge queue | GitHub protections/maintainer | `pending` | Capture exact-head workflow URLs and conclusions after push; local green is insufficient. |
| Signing and release provenance | Release candidate, SBOM, checksums, signatures and attestations | Release signer/maintainer | `pending` | Generate and sign the release packet through the approved release workflow. |
| Registry/publication | PyPI/conda, R registry, Julia General/JLL, Rust and publication lanes | Respective registry maintainers/editors | `pending` | Record submission/indexing/approval receipts; prepared manifests are not publication. |
| Parent and child issue closure | #313, #314–#323 and historical #416 | Maintainer/GitHub authority | `pending` | Reconcile live issue/Project events and close only after repository acceptance plus external receipts. |

## Decision boundary

No user decision is required to record these gates. The recommended policy is
to keep all non-repository authorities `pending` or `fail_closed` until their
receipts are available. Options to accelerate promotion, waive scientific
review, or close parent issues without receipts are rejected by the track
contract.
