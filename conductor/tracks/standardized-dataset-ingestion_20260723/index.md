# Standardized Dataset Ingestion

Status: in progress; PR #334 establishes the tested baseline implementation,
PR #477 carries normalized-input provenance through the existing runtime, and
PR #494 explicitly rejects unsupported Croissant archives and transformations.
The remaining published conformance, security, SDK, and cross-domain acceptance
criteria stay active. These are merged increments, not track closeout evidence.

Local evidence on the `codex/sdk-dataframe-worked-examples` branch extends the
Phase 8–9 DataFrame SDK consumer contracts and routes both business reference
cases through the public adapter with `allow_copy=False`. It is partial local
evidence only: the related plan tasks, hosted matrix, and external GitHub
acceptance checks remain open.

GitHub parent issue:
[#325](https://github.com/edithatogo/voiage/issues/325)

GitHub Project:
[VOP–VOIAGE Conductor Roadmap](https://github.com/users/edithatogo/projects/28)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)
