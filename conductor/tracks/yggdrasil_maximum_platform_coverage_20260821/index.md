# Yggdrasil Maximum Platform Coverage

Status: in progress — specification and implementation plan approved by the
repository owner on 2026-08-21; Phase 1 contract work is starting.
No external recipe change, expanded hosted run, upstream merge, JLL generation,
or Julia registry outcome is claimed by track initialization.

GitHub owner issue: [#555](https://github.com/edithatogo/voiage/issues/555)

External recipe: [JuliaPackaging/Yggdrasil PR #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)

## Scope boundary

The track maximizes attempted BinaryBuilder coverage by starting from
`supported_platforms()` and adding only narrow, evidenced negative filters. It
separates compilation, product, ABI-smoke, and runtime evidence. Yggdrasil
approval and merge, JLL generation, downstream Julia execution, General
registration, and indexing remain external gates.
