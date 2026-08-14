# Phase 10 reconciliation evidence — 2026-07-31

This is an additive reconciliation record for the range after the earlier
Phase 10 record.  It does not rewrite the legacy ledger or turn a merged pull
request into evidence that an unimplemented acceptance criterion is complete.

## Collection method and limits

At `aa500d733d0de86e2c6051461249a64f4a5dc585`, the maintainer collected
GitHub REST pull-request records for #639--#690, `git diff-tree` artifact
lists for each merged ingestion increment, and the merge-commit check-run
inventory.  The complete project-item query was:

```sh
gh project item-list 28 --owner edithatogo --format json --limit 500
```

All ten track issues (#325--#333, #467, and #468) are present in Project 28
and reported `In Progress` at collection time.  Project status is a planning
field, not an acceptance-criterion verdict, so this snapshot is deliberately
not used to close an issue or the track.

The check-run inventory is retained as provenance only.  Merge is direct
evidence that GitHub accepted the PR under the then-current protected rules,
but check-run history can also contain skipped jobs and post-merge release or
dependency-submission failures.  This record therefore does not assert a
retrospective all-green matrix from the inventory; the corresponding PR and
commit remain the authoritative hosted-check reference.

## Merged ingestion increments

Each row records a GitHub-merged PR, its exact squash merge commit, and
representative changed artifacts found by `git diff-tree -r`.  The complete
artifact list remains available from that immutable commit.

| PR | Merge commit | Evidence scope | Representative artifacts |
| --- | --- | --- | --- |
| #639 | `928a7281069f9c2590d71e0a80e07b366dc1617e` | fixture-evidence reconciliation | `conductor/github-cross-references.json`; track `plan.md` |
| #640 | `46f4f887f9041ee32cc19610b577d42211bf9d96` | malformed Croissant collections | Croissant unsupported fixtures |
| #641 | `d0bd9667a9b65d3bd7fe633cfcf9ddc4b32a8d6f` | malformed Frictionless resource | Frictionless unsupported fixture; fixture tests |
| #642 | `28b2609c37f0b2451093a80bc0bbd0ee118595e7` | non-object descriptor rejection | Croissant and Frictionless unsupported fixtures |
| #643 | `46ef787374b9a6d376764878849cab61636a38a2` | lazy built-in provider exports | `voiage/ingestion/__init__.py`; provider tests |
| #644 | `64ef08e35f7d51d5519377095b8690992a0fa73a` | provider inspection evidence | CLI ingestion tests |
| #645 | `e6321b4f83f4d8b19f539c1ad0911cf16864d110` | clean-install record | track `plan.md` |
| #646 | `e63be6206924a08b480517732e7ee70f721895b0` | receipt parity | conformance tests |
| #647 | `cf90f32c8f4652aa9650cd98fd0dcdd133917ae1` | fresh-process Arrow round trip | conformance tests |
| #648 | `623fa73ba1efc2edc8f36a2c3c1a237a9d279453` | supported profile matrix | standardized-ingestion documentation |
| #649 | `0a297f7d0c543e6b836c04db2444c5575ee1aa73` | deterministic fixture manifests | CLI and conformance tests |
| #658 | `c59966923bc236ee886fa82036c81f5f91b838b8` | supported descriptor workflow | standardized-ingestion documentation |
| #659 | `1fad7f224bbefd6f2eae56dee627c078aab029ce` | reference-case CLI walkthrough | CLI ingestion tests |
| #660 | `dba2886a932bc6c2987cdb90bf1f5d9fafb79430` | DataFrame conversion diagnostics | DataFrame SDK tests; standardized-ingestion documentation |
| #661 | `6e609c69766f957fb83e4cb6ff731aba7980f842` | cache namespace containment | `voiage/ingestion/base.py`; source-policy tests |
| #662 | `25722c32473874be48d0fac477fb11c0876c43af` | conformance CI lane | `tox.ini`; CI workflow |
| #663 | `0c1bde9b659cce8a17fd57dec40cd8d7e1712314` | metadata-only inspection | `voiage/ingestion/registry.py`; provider tests |
| #665 | `f36111d7c94087fc084b0b631a1df6d87f07a663` | all-surface reference cases | reference-case example and tests |
| #666 | `c84086d578adfc8d2cc1ecb0d71fc0b5946d4daa` | ambiguous Frictionless field rejection | provider tests; track evidence |
| #667 | `a51d10b4be764a733b360b76963afed43e586936` | Croissant context arrays | Croissant valid/unsupported fixtures |
| #668 | `5a5ddcf8de966b9d9f3ea014d3a696004cabdb58` | explicit CLI source roots | `voiage/ingestion/cli.py`; CLI tests |
| #669 | `f8d6362d1d5a5bc965679726e8158dcdce0fcb6e` | cross-provider mapping properties | conformance and provider tests |
| #670 | `96c1101464583a3fb093541cfed684f0888d92d5` | hard-link cache rejection | `voiage/ingestion/base.py`; cache tests |
| #683 | `12ed932b0d27adde075e0f0032330d87c1b5f727` | non-materializing CLI inspection | `voiage/ingestion/cli.py`; CLI tests |
| #684 | `27e222c3dee9db3afb2544bb4b59ca4ffa7065c0` | pinned normalized identities | standardized-ingestion manifests |
| #685 | `eaf528428a5476528ac99d86dc0507dca3afa3a1` | Croissant context corpus | Croissant fixtures; documentation |
| #686 | `5c3f9d72048b1a10a39d9c1b580d2d7b07e1deeb` | Frictionless fixture corpus | Frictionless fixtures; fixture tests |
| #687 | `d3a4882494f8caa3fe3dee2f4f859b80514f64c8` | DataFrame consumer diagnostics | DataFrame SDK tests; documentation |
| #689 | `2317c1ca7f1e652eb822ac92b80a8a3e2ca609fe` | source-policy assurance and ledger repair | source policy; ledger repair receipt |
| #690 | `aa500d733d0de86e2c6051461249a64f4a5dc585` | ML, engineering, and business cases | cross-domain examples, manifests, tests |

## Range exclusions

- #650 was closed without merge and is excluded.
- #651--#654 are JOSS/review work, and #675, #677, #678, and #688 are release
  work; none is used as standardized-ingestion implementation evidence.
- #655--#657, #671--#674, and #680--#682 have no pull-request resource in the
  GitHub REST collection and are recorded as absent rather than guessed.
- #676 and #679 are open, unrelated estimation/governance PRs and are excluded.

## Acceptance conclusion

The mapping supports the repository-owned partial increments recorded for
P4--P9 and makes their PR provenance available in the central cross-reference
manifest.  It does **not** complete P4 live authoritative probes, all P4
format/profile coverage, P5 parser-differential acceptance, P6/P7 final phase
checkpoints, remote/DNS/redirect/archive/mutable-live policy evidence, or the
whole-track review/archive requirements.  The parent issue, every listed
sub-issue, Project 28 items, and this Conductor track remain active.
