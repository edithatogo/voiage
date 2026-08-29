# PR #855 closeout plan

This plan addresses the stale/conflicting scientific-review implementation branch
without treating hosted automation as scientific approval.

## Objective

Rebind PR #855 to the current `main` tip, repair deterministic Conductor
normalization drift, validate the exact candidate locally, and obtain a fresh
hosted-check result before any merge decision.

## Execution sequence

- [x] Freeze the PR identity, branch, failure evidence, and current `main`
  revision in the review packet.
- [x] Rebase `codex/scientific-review-implementation-20260802` onto `origin/main`
  with conflicts resolved in favour of the PR's scientific-review changes.
- [x] Repair the normalizer so archived historical follow-up markers and
  documentation-only contract comments are idempotent and do not become active
  pending work.
- [x] Run focused scientific-review and Conductor-normalization tests locally.
- [x] Force-push the rebased exact head and bind hosted checks to that head.
- [ ] Require all exact-head unit, compatibility, coverage, security and
  packaging checks to finish successfully.
- [ ] Merge only after the exact-head checks are green and the maintainer has
  confirmed that scientific, parity, promotion, release and publication gates
  remain separately pending.

## Contingencies and decision rules

1. If only normalization drift fails, apply a deterministic normalizer/test
   repair and repeat the exact-head cycle.
2. If scientific-review assertions fail, preserve the failing evidence and
   repair the owning review-contract slice; do not weaken assertions or mark
   scientific approval complete.
3. If hosted infrastructure is unavailable, retain the PR open with a pending
   hosted gate and record the run URL; local green does not substitute for CI.
4. If the rebased diff materially changes candidate semantics, stop and require
   a new candidate freeze and panel delta review before merging.

## Acceptance boundary

Repository integration is complete only when the exact rebased head is green.
This plan does not authorize scientific approval, installed-language parity,
stable promotion, release, registry/publication acceptance, or parent-issue
closure.
