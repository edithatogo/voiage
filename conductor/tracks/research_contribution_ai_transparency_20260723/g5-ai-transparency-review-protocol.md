# G5 AI-transparency conformance and pathological review protocol

Run this protocol offline against deterministic fixtures before accepting any
AI-assistance record or public disclosure. Each case records the input hash,
diagnostic, output hash and tool version. A local pass does not certify
authorship, consent, scientific review, release or publication.

| ID | Mutation | Required disposition |
|---|---|---|
| AI-01 | Missing or extra ledger field | Reject with field-path diagnostic. |
| AI-02 | Unknown provider/model represented as a guessed value | Reject; unknown identity must be `null`. |
| AI-03 | Raw prompt, chain-of-thought, secret, token or confidential text | Reject and prove redaction; value must not appear in output/logs. |
| AI-04 | Duplicate event ID or altered prior event | Reject hash-chain replay; preserve original evidence. |
| AI-05 | Wrong release tag/commit or artifact hash | Reject release binding; no public claim. |
| AI-06 | Empty purpose, scope, limitations or verification method | Reject as incomplete provenance. |
| AI-07 | `disposition=accepted` without human reviewer identity | Reject; AI cannot self-approve. |
| AI-08 | AI listed as author or CRediT contributor | Reject claim and emit authorship-boundary diagnostic. |
| AI-09 | Non-canonical JSON ordering, timestamp or Unicode normalization | Reject deterministic serialization mismatch. |
| AI-10 | Manuscript disclosure differs from release ledger | Reject synchronization check and identify both hashes. |
| AI-11 | Path traversal or symlink in referenced artifact scope | Reject before reading outside the repository fixture root. |
| AI-12 | Replayed valid ledger with same release and seed | Require byte-identical canonical output and event hash. |

The review manifest, fixtures and results are evidence artifacts. Any failing
case blocks maturity promotion until remediated; it is never converted into a
warning or silently excluded.
