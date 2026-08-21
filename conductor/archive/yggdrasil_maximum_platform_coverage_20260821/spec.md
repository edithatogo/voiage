# Yggdrasil Maximum Platform Coverage Specification

## Overview

Replace the fixed seven-target allowlist in the submitted `voiage_ffi`
Yggdrasil recipe with an inclusive, evidence-governed platform contract. The
recipe starts from BinaryBuilder's current `supported_platforms()` universe and
removes only targets that have a precise, reproducible reason they cannot build
or ship the `voiage_ffi` shared library.

This track owns the repository-side contract, validator, tests, evidence
receipts, and the maintainer-controlled update to the existing Yggdrasil pull
request. Yggdrasil review and merge, JLL generation, Julia General registration,
and registry indexing remain external outcomes.

## Authoritative inputs

| Input | Authority and pinned revision |
| --- | --- |
| VOIAGE repository state | `edithatogo/voiage` commit `658a89e4525300f19b6a0e4ce0d65d0c6ce64bee` |
| Released native source | `edithatogo/voiage` commit `964a0fc334ece9509387cd07d43776adf38be240`, as pinned by the submitted recipe |
| Repository recipe mirror | `packaging/yggdrasil/V/voiage_ffi/build_tarballs.jl` |
| External recipe candidate | [JuliaPackaging/Yggdrasil PR #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292), head `2528e2efb90e4197924d45c98873ca5cdb1a9d42` based on `5059405e0e9ecced8fb1619baff0cfc6c5478742` and observed 2026-08-21 |
| Yggdrasil review requirement | [Review comment `discussion_r3658212619`](https://github.com/JuliaPackaging/Yggdrasil/pull/14292#discussion_r3658212619), answered by [reply `discussion_r3831637213`](https://github.com/JuliaPackaging/Yggdrasil/pull/14292#discussion_r3831637213); its requested policy remains implemented at candidate head `2528e2efb90e4197924d45c98873ca5cdb1a9d42` |
| Yggdrasil platform catalogue checkpoint | `JuliaPackaging/Yggdrasil` `master` commit `5059405e0e9ecced8fb1619baff0cfc6c5478742`, observed 2026-08-21 |
| BinaryBuilder platform catalogue source | `BinaryBuilderBase` commit `76c4aab80ad5019af59af0f42e5669109cd5194b` and tree `38ac28858e80c575fc2ff3c7ac73982459c4482d`, as resolved by the Yggdrasil checkpoint manifest; [`supported_platforms()` source](https://github.com/JuliaPackaging/BinaryBuilderBase.jl/blob/76c4aab80ad5019af59af0f42e5669109cd5194b/src/Rootfs.jl#L778-L832) |
| Existing release-candidate receipt | `conductor/archive/research_software_registry_readiness_20260721/release-2.1.0-registry-candidate-receipt-20260821.json` |
| Julia delivery contract | `docs/release/binding-submission-checklist.md` and GitHub issue [#555](https://github.com/edithatogo/voiage/issues/555) |

If a later Yggdrasil platform catalogue or maintainer instruction conflicts
with this checkpoint, the newer authoritative upstream state must be captured
as a fresh evidence receipt and the contract updated before filtering.

## Requirements

### R1 — Inclusive platform universe

The candidate recipe must derive `platforms` from `supported_platforms()`.
Architectures, operating systems, C libraries, or ABI variants must not be
enumerated as a positive allowlist merely because they were the previously
tested set.

### R2 — Minimal initial negative filter

The first maximum-coverage attempt must exclude only the two upstream-identified
Rust toolchain gaps:

1. FreeBSD on `aarch64`, using the narrow predicate
   `Sys.isfreebsd(p) && arch(p) == "aarch64"`.
2. Every `riscv64` platform, using `arch(p) == "riscv64"`.

Each filter must include an adjacent reason. Windows `i686`, Linux `i686`, ARM,
PowerPC, FreeBSD `x86_64`, musl, and other catalogue members must be attempted
unless evidence from the maximum-coverage build justifies a narrower exclusion.

### R3 — Machine-readable coverage contract

The repository must contain a versioned JSON Schema and manifest describing:

- the Yggdrasil catalogue revision and recipe candidate revision;
- the exact platform universe observed for that revision;
- every included and excluded platform;
- one lifecycle status per platform: `pending`, `building`, `passed`,
  `failed_actionable`, `failed_transient`, `excluded_upstream`, or
  `excluded_evidenced`;
- build, product, ABI-smoke, and runtime-validation states as separate fields;
- for each exclusion, its exact Julia predicate, normalized reason category,
  primary evidence locator, observation timestamp, and reconsideration trigger;
- aggregate counts that reconcile exactly with the platform records.

The validator must fail closed on an unclassified catalogue platform, an
included platform without a lifecycle record, a broad or unexplained exclusion,
duplicate platform identity, stale aggregate counts, or a claim stronger than
its evidence.

### R4 — Failure classification and remediation order

Every failed build must be classified before changing the filter:

1. Recipe, source, compiler-flag, linker, product-path, or ABI problem: fix the
   recipe and retain the platform.
2. Transient runner, cache, or network problem: rerun without excluding.
3. Missing upstream Rust target or toolchain: record primary upstream or
   reproducible hosted evidence, then add the narrowest predicate.
4. Unsupported shared-library output: investigate target-specific linking and
   artifact installation before exclusion.
5. Fundamental `voiage_ffi` architecture limitation: document the limitation,
   test the negative predicate, and require an explicit reconsideration trigger.

No failure may be converted directly into a platform exclusion without this
triage.

### R5 — Build and product integrity

For every included platform the recipe must continue to:

- build the locked `voiage-ffi` Cargo package;
- preserve the musl shared-linking adjustment where required;
- install the correctly named shared library;
- declare `LibraryProduct("libvoiage_ffi", :libvoiage_ffi)`; and
- avoid claiming runtime or numerical validation from compilation alone.

Where the generated artifact can execute on available infrastructure, the
evidence must also verify the exported C ABI, version agreement, and at least
one deterministic EVPI reference case. Non-executable cross targets must be
reported as build/product validated only.

### R6 — Exact-head evidence and reproducibility

Every hosted run receipt must bind the Yggdrasil PR number, candidate head,
base revision, Buildkite build, exact platform statuses, recipe digest, source
revision, and observation time. Superseded and failed runs remain historical
evidence; a later success must not rewrite them.

### R7 — Repository and external-state reconciliation

The final repository checkpoint must reconcile the platform manifest,
repository recipe mirror, issue #555, issue #614 where applicable, the archived
registry-readiness handoff, and Yggdrasil PR #14292 without treating an open or
green external PR as merged. A generated JLL, Julia General registration, and
registry indexing may be recorded only from authoritative external receipts.

## Acceptance criteria

- **AC1:** A versioned schema, manifest, and validator enforce R1–R6 and pass
  positive, negative, malformed, and aggregate-reconciliation tests.
- **AC2:** The repository recipe mirror and exact external PR candidate use
  `supported_platforms()` with only the two initial upstream exclusions before
  the first expanded hosted run.
- **AC3:** Every platform in the pinned catalogue is included or excluded once,
  and every additional exclusion is the narrowest reproducible predicate with
  evidence and a reconsideration trigger.
- **AC4:** All included platforms reach a terminal hosted state; actionable
  recipe failures are repaired and transient failures are rerun rather than
  filtered.
- **AC5:** Product and ABI/runtime claims are no stronger than the exact
  per-platform evidence, and executable targets complete the declared smoke
  checks when the generated artifact is available.
- **AC6:** Focused tests, full Conductor validation, cross-reference validation,
  repository harness, relevant tox environments, and diff hygiene pass at the
  exact repository candidate.
- **AC7:** GitHub and Conductor records link the exact PR/run receipts and
  preserve Yggdrasil merge, JLL generation, Julia General, and indexing as
  separate external gates.

## Non-functional constraints

- The policy must maximize attempted coverage, not the number of green checks
  obtained by pre-emptive filtering.
- Evidence must be deterministic, content-addressed where stored locally, and
  append-only when superseded.
- Filters must remain readable Julia predicates with adjacent rationale.
- The contract must be maintainable when `supported_platforms()` changes;
  catalogue drift must fail validation until newly observed platforms are
  classified.
- No credential, authenticated registry action, or external acceptance claim is
  stored in repository evidence.

## External gates

The following do not block repository-owned contract completion, but must
remain visible and pending until authoritative evidence exists:

1. Yggdrasil maintainer approval and merge of PR #14292.
2. BinaryBuilder generation and registration of `voiage_ffi_jll` artifacts.
3. Downstream clean-depot Julia smoke tests using the registered JLL.
4. Registrator submission, Julia General review/merge, and registry indexing.

## Out of scope

- Adding a new VOI method or changing stable numerical semantics.
- Claiming native execution on a cross target from compilation alone.
- Excluding platforms solely because another package excludes them.
- Broadening the Julia source package API before its JLL dependency exists.
- Merging the upstream PR or triggering downstream publication without the
  upstream destination's own authority and evidence.
