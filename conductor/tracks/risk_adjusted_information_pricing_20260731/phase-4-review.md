# Phase 4 automated review

## Scope

Reviewed the Phase 4 diff from checkpoint `79c312f6` through U17, covering the
CLI and VoC presentation, provisional exports, documentation and examples,
language capabilities, frontier discovery and promotion, canonical and GitHub
governance, and the frozen v1 boundary.

## Findings and resolutions

- **High — delivery PR ownership:** draft PR #712 was initially inserted under
  an adjacent programme track rather than the owning #595 track. The manifest
  now agrees exactly with the owning track metadata, with a regression test.
- **High — bundled fixture validation:** the first registry extension checked
  existence and hashes but was fail-open to duplicate or escaping paths and did
  not validate payload schemas. The validator now applies the registry schema,
  unique and contained paths, lowercase SHA-256, request/result/reference
  schemas, normalized resolved-path duplicate detection (including symlink
  aliases), and negative mutation tests.
- **Medium — Rust shared-fixture evidence:** the first capability test proved
  only that the Rust evidence file existed. Rust now consumes both committed
  normative reference fixtures directly and compares all five measures.
- **Medium — CLI failed-root presentation:** the first human-readable CLI path
  rendered every null value as unavailable. Failed measures now exit through
  the error boundary with their termination reason and diagnostic reference;
  genuinely unavailable measures retain their distinct state.
- **Governance baseline:** the newly materialized #595 track is now included in
  the active/post-v1 programme baseline and its contract tests.

No stable-v1 ABI expansion or duplicate VoC numerical kernel was found.

## Validation

- Rust expected-utility reference and pathology suites: 13 passed.
- Strict Rust clippy for `voiage-numerics`: passed with warnings denied.
- Focused Python contract, runtime, CLI, capability, frontier, v1 baseline and
  Conductor governance suite: 93 passed.
- Frontier fixture validator: passed, including the bundled family.
- Full Conductor validator: 144 tracks, zero errors, zero warnings.
- Initial full repository run: 2,548 passed, 15 skipped, one governance
  baseline failure, 91.80% coverage. The baseline failure is resolved above;
  the final full rerun is recorded by U17 evidence.

## Remaining external gates

Scientific-design review, exact-head hosted checks, stable promotion, merge,
release and issue closure remain separate. The draft PR must not be described
as merge-ready until its conflict and hosted-check state are resolved.
