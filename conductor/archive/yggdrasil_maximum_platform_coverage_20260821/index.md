# Yggdrasil Maximum Platform Coverage

Status: repository work complete and archived; exact-head hosted refresh pending. The exact Yggdrasil
candidate passed all 15 included platforms in Buildkite 31972; three narrowly
evidenced exclusions remain. All 15 product archives have integrity evidence,
and two runnable macOS artifacts have ABI and numerical smoke evidence.

GitHub owner issue: [#555](https://github.com/edithatogo/voiage/issues/555)

External recipe: [JuliaPackaging/Yggdrasil PR #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292)

Voiage implementation: [PR #999](https://github.com/edithatogo/voiage/pull/999)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)
- [Phase 1 automated review](./phase-1-review-20260821.md)
- [Phase 2 automated review](./phase-2-review-20260821.md)
- [Phase 3 automated review](./phase-3-review-20260821.md)
- [Phase 4 product and ABI evidence](./phase-4-product-abi-evidence-20260821.json)
- [Phase 4 automated review](./phase-4-review-20260821.md)
- [Phase 5 final automated review](./phase-5-review-20260822.md)
- [Rebase and validation receipt](./phase-5-rebase-validation-20260822.json)
- [Registry-readiness successor](./registry-readiness-yggdrasil-successor-20260821.json)

## Scope boundary

The track maximizes attempted BinaryBuilder coverage by starting from
`supported_platforms()` and adding only narrow, evidenced negative filters. It
separates compilation, product, ABI-smoke, and runtime evidence. Yggdrasil
approval and merge, JLL generation, downstream Julia execution, General
registration, and indexing remain external gates.
