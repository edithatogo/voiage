# Specification: Remaining Backlog Delivery

## Objective and authorization

Complete the maintainer's 31 August 2026 instruction to address remaining work,
use parallel agents, merge all eligible PRs, address every GitHub issue, and
prune retired branches and worktrees. This track records that existing scope;
it does not replace individual issues' acceptance criteria or reopen completed
historical tracks. Issue #1053 coordinates delivery.

## Authoritative inputs

- `AGENTS.md`, `CONTRIBUTING.md`, `conductor/workflow.md` and
  `.github/workflows/` at baseline `27488e817238d6fe63016f2f5a5b15f91b1acda7`.
- `todo.md`, `roadmap.md`, `specs/submission-readiness/targets.json` and
  `conductor/tracks/v2_2_release_and_venue_submissions_20260830/` at that baseline.
- Current issue acceptance criteria and hosted PR/check state at
  [voiage](https://github.com/edithatogo/voiage/issues).
- Protected `main` ruleset 18831614: current-base checks, signed commits,
  linear history, resolved threads and zero required independent approvals.

## Acceptance criteria

1. **AC1 — Inventory:** audit all open PRs and issues, retaining source heads,
   current acceptance criteria, delivery state and unresolved gates.
2. **AC2 — Technical repairs:** complete dependency delivery and Codecov
   controls (#656), opt-in test acceleration (#1028), HPC packaging (#1025),
   and remaining repository-owned R, governance and venue preparation.
   Preserve historical evidence and full CI, numerical and coverage gates.
3. **AC3 — Delivery:** self-review each diff, run full local verification,
   resolve hosted failures and merge each eligible PR sequentially against
   its reviewed head. Superseded or rejected historical PRs remain historical.
4. **AC4 — Issue reconciliation:** update every open issue against its actual
   criteria. Close only completed work; keep human and external outcomes open
   when their required evidence is absent.
5. **AC5 — Recovery and cleanup:** preserve unique work, Git notes, backup refs
   and every stash reflog commit in a verified, independently restored archive
   before pruning. Recheck current branch tips and open PRs; use exact-SHA
   deletion leases. Keep maintained release branches and active worktrees.
6. **AC6 — Final audit:** verify live PR, issue, branch and worktree inventories.
   Do not claim the overall goal complete while required work remains.

## External gates and exclusions

Personal pyOpenSci declarations are already confirmed. Survey completion and
human-written submission text remain actual human-action requirements. Follow
the authorized pyOpenSci-first, eligible JOSS, then deferred arXiv sequence.
Scientific/domain review, genuine non-author use, registry curation, fiscal
hosting, badges, editorial acceptance and indexing require their own evidence.
Automated agents cannot supply human independence or attest actions not taken.
Do not send new outreach, submit upstream packaging applications, make new
authorship/category/license choices, publish another release, or weaken
protected controls merely to close an issue.
