# G13 programme and external-gate reconciliation

## Local authoritative reconciliation

The root programme is issue **#313** in Project 28. Its native workstreams are
#314–#323; historical programme evidence #416 is retained. The local Conductor
registry, roadmap and todo files agree on the track-to-issue mapping and do not
convert GitHub/Project status into implementation evidence.

| Issue | Workstream | Local Conductor state | Closure interpretation |
|---:|---|---|---|
| #314 | Method census and contracts | In progress | Repository reconciliation exists; review and closure evidence remain. |
| #315 | External VOI feature parity | New | Specification/plan materialized; delivery and review remain. |
| #316 | Stable Rust VOI core | In progress | Rust work exists; stable promotion and external evidence remain. |
| #317 | Value of Perspective | New | Repository work and panel boundaries remain separate from scientific approval. |
| #318 | Supported frontier methods | In progress | Child evidence is tracked; scientific, parity, release and closure gates remain. |
| #319 | ML/LLM/agent VOI | New | Specification/plan materialized; implementation and review remain. |
| #320 | Polyglot ABI and binding parity | New | Binding contracts exist; installed parity and registry gates remain. |
| #321 | Datasets and worked examples | In progress | Rights-governed packs and hosted evidence remain. |
| #322 | Quality/release automation | New | Release and registry evidence remain. |
| #323 | Research contribution and AI transparency | New | Accountable authorship and publication evidence remain. |
| #416 | Historical v1.0 evidence | Completed historical record | Preserved as evidence; not used to close #313. |

## Release and destination lanes

The release closure state remains **blocked/not_checked** until authoritative
receipts exist for Python/conda-forge, R/CRAN, Julia BinaryBuilder/General,
Rust crates, Software Heritage, RRID, JOSS and arXiv. Local manifests,
checksums, SBOMs and provenance can establish `ready`, but not `submitted`,
`published`, `indexed` or `approved`.

## Parent-closure rule

Do not close #313 until every workstream has repository acceptance evidence and
its remaining scientific, maintainer, hosted, signing, registry, publication
and issue-closure gates are explicitly receipted. No GitHub mutation was
performed by this reconciliation.
