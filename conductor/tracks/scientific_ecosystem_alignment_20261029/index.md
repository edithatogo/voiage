# voiage: scientific ecosystem alignment

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Coordinator](https://github.com/edithatogo/repository-standards/tree/7874f05132447c77f681fbca7ad5394066564aaf/conductor/tracks/scientific_ecosystem_alignment_20261029)

- [GitHub issue](https://github.com/edithatogo/voiage/issues/1221)

## Non-HPC frontier follow-on (8 November 2026)

- [Programme contract and entry point](./frontier_20261108/README.md)
- [Ordered implementation packets](./frontier_20261108/plan.md)
- [Verification datasets and feature matrix](./frontier_20261108/verification-matrix.md)
- [Research, repository alignment and publication](./frontier_20261108/research-publication.md)
- [Primary sources and verification boundaries](./frontier_20261108/sources.md)
- [Packet dependencies](./frontier_20261108/packets.json)

These are pending follow-on plans, not implementation, numerical acceptance or a
new release. Existing receipts, integrator ownership and HPC deferral are retained.

## Strategic information integration — 8 November 2026

- [Owner-scoped roadmap and acceptance cases](strategic-information-integration.md).
  Planning only; existing implementation statuses, pins and release gates remain unchanged.

### Canonical provider correction: Paignia

The maintainer identified [Paignia](https://github.com/edithatogo/paignia) as the
reusable Rust game-library owner. This supersedes any reading of the earlier
incubator/extraction descriptions as an undecided repository destination.
`new-drug-reimbursement-game` remains the historical seed and a domain application;
Voiage retains information-value calculations, UOGTO semantics, and Kairos optional
execution. The [provider-side handoff](https://github.com/edithatogo/paignia/pull/1)
records the corresponding integration requirements.

The [planning manifest](strategic-information-manifest.json) now names Paignia.
Its bootstrap revision is an observation, not a dependency pin or qualification.
T17 identifiers remain historical obligations: map them to actual Paignia source
and acceptance evidence before provider admission. Do not infer crate releases,
completed extraction or solver support from the repository name. The two-application
rule qualifies static packages/consumers, not creation of this existing repository.
SI01–SI04, their eight cases, pending states and core-release independence are unchanged.
