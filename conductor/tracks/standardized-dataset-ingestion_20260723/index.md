# Standardized Dataset Ingestion

Status: in progress. The additive [2026-07-31 Phase 10 reconciliation record]
(./p10-reconciliation-20260731.md) maps 30 merged ingestion increments in
PRs #639–#690 to exact commits, artifacts, and hosted-check provenance while
recording scope exclusions. Issues #325–#333, #467, and #468 remain open, and
their Project 28 items remain `In Progress`; neither fact is treated as
closeout evidence. The pre-repair ledger is preserved byte-for-byte and the
current valid chain is bound by [the integrity-repair record]
(./evidence-integrity-repair-20260731.md). These are increments, not track
closeout evidence.

The approved 2026-08-01 strict-local completion boundary retains deterministic,
offline Croissant and Frictionless support in this track. Controlled live
interoperability and any general remote-ingestion policy are successor work in
[#752](https://github.com/edithatogo/voiage/issues/752) and
[#753](https://github.com/edithatogo/voiage/issues/753); neither is evidence
that this track supports live or remote sources.

GitHub parent issue:
[#325](https://github.com/edithatogo/voiage/issues/325)

GitHub Project:
[VOP–VOIAGE Conductor Roadmap](https://github.com/users/edithatogo/projects/28)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)
- [Pre-repair Evidence](./evidence.pre-integrity-repair-20260731.jsonl)
- [Integrity Repair](./evidence-integrity-repair-20260731.md)
- [P10 Reconciliation](./p10-reconciliation-20260731.md)
