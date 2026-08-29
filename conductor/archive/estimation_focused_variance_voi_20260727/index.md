# Estimation-Focused Variance-Reduction VOI

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [MoSCoW Requirements](./requirements.md)
- [Mermaid Design](./design.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)
- [Governed delivery closeout](./delivery-closeout-20260801.md)
- [GitHub issue #619](https://github.com/edithatogo/voiage/issues/619)
- Subissues:
  [#671](https://github.com/edithatogo/voiage/issues/671),
  [#672](https://github.com/edithatogo/voiage/issues/672),
  [#673](https://github.com/edithatogo/voiage/issues/673), and
  [#674](https://github.com/edithatogo/voiage/issues/674)
- [Frontier parent #318](https://github.com/edithatogo/voiage/issues/318)
- [Programme #313](https://github.com/edithatogo/voiage/issues/313)
- [Project 28](https://github.com/users/edithatogo/projects/28)
- [Implementation PR #676](https://github.com/edithatogo/voiage/pull/676)
- [Canonical C16 implementation sync PR #64](https://github.com/edithatogo/vop_poc_nz/pull/64)
- [Scientific remediation issue #843](https://github.com/edithatogo/voiage/issues/843)

Status: experimental scalar Rust/Python implementation, assurance and user
surfaces passed PR #676's 65 exact-head contexts (60 successes, four governed
skips and one neutral CodeQL aggregation) and merged as `9495fc3f`. Canonical
sync PR #64 also passed its 16-context matrix and merged as `cedc6fbb`.
Repository-delivery subissues #671--#674 are eligible for closure. Planned
Phase 7 E19–E23 scientific remediation/re-review, vector covariance, stable
promotion, release, parent #619 closure and umbrella #318 closure remain
separate gates, so the track remains active.

Status: superseded on 2026-08-29 by `pre_submission_comprehensive_hardening_20260829`. Historical implementation and evidence remain preserved; pending work migrated without a completion claim.
