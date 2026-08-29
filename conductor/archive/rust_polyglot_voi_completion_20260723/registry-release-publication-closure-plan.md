# Registry, release, publication and parent-issue closure plan

## State model

Every destination and parent issue uses distinct states: `ready`, `submitted`,
`published`, `indexed`, `approved`, `blocked`, `rejected`, and `not_checked`.
Repository artifacts may establish `ready`; only an authoritative registry,
release, journal, archive or GitHub response can establish the later states.

## Ordered workflow

1. **Reconcile:** validate `specs/submission-readiness/targets.json`, the
   binding checklist, registry audit snapshot, release manifests and Conductor
   cross-references. Every open destination must have an owner, issue, next
   action and external evidence field.
2. **Release candidate:** bind source revision, versions, SBOM, provenance,
   checksums, signatures, clean-install results and compatibility matrix. A
   release is not complete from a local build alone.
3. **Registry lanes:** maintain separate Python/conda-forge, R/CRAN,
   Julia/BinaryBuilder/General, Rust/crates.io, Software Heritage, RRID and
   JOSS/arXiv lanes. Record submission receipts and review outcomes without
   credentials or irreversible uploads from Conductor automation.
4. **Publication:** require human author/editor approval for manuscript,
   category, licence, authorship, affiliations, AI disclosure and final
   submission. Drafts and prepared packets remain non-submissions.
5. **Parent closure:** close child issues only when their repository acceptance
   evidence is complete. Close parent #313 only after all workstreams have
   reconciled evidence and explicit external gates; do not close it because
   Project status, PR merge or release exists.
6. **Final receipt:** record the exact authority/portal, response URL or issue
   event, timestamp, artifact hashes, state transition and next action. If
   evidence is unavailable, retain `not_checked` or `blocked`.

## Approval boundaries

The maintainer may approve repository promotion and issue closure. Registry
maintainers, journal editors, archive curators, release signers and authors
retain authority over their respective external decisions. Conductor panels
prepare evidence and recommendations but cannot substitute for those actors.
