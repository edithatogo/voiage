# Remaining blockers closeout plan

## Recommendation

Use **Option A — dependency-ordered closeout with hosted evidence receipts**.
It maximizes autonomous progress while preserving the distinction between
repository readiness and external authorization. No gate is marked complete
from a local proxy, a missing receipt, or an inferred GitHub state.

## Ordered workplan

| Order | Gate | Action | Evidence required | Contingency |
|---:|---|---|---|---|
| 1 | G4 | Run contract, schema, cross-reference and full Conductor validators. | Versioned command results and artifact hashes. | If a validator fails, apply at most two scoped fix/re-run loops; retain the failure if it is a baseline issue. |
| 2 | G5 | Complete negative/conformance/property evidence for the accepted Rust/polyglot surfaces. | Test IDs, fixture hash, failure diagnostics and tolerance results. | Record a reviewed protocol for any runtime unavailable locally; do not fabricate tests. |
| 3 | G7 | Consolidate rights, privacy, scientific, practitioner and external-gate dispositions. | Gate ledger with owner, authority, state and next action. | Keep `pending`/`not_checked` when accountable evidence is human-owned. |
| 4 | G10 | Run Rust/Python/R/Julia from clean installed environments against one immutable fixture hash. | Toolchain, platform, install/test commands, output/diagnostic hashes, ABI/layout and unsupported-capability results. | Current local fallback is the Homebrew Python/Rust FFI/Julia setup; R package mock tests remain development-context-only. |
| 5 | G12 | Run tox lint, harness, typecheck, frontier, version-sync, docs and focused tests. | Exact revision, exit status and redacted summaries. | Fix tox configuration/dependency defects in scope; otherwise record environment-specific failures. |
| 6 | G13 | Refresh GitHub/Project child and parent states from authoritative receipts. | Issue/Project URLs, event timestamps and receipts. | If GraphQL or account access is unavailable, retain local reconciliation and mark live states `not_checked`. |
| 7 | G14 | Bind the final local packet and exact-head hosted checks to the release candidate. | Release-candidate commit, workflow run IDs, all required check conclusions. | Rebase/re-run when the head changes; never treat stale checks as current. |
| 8 | G15 | Record repository completion separately from merge, release, publication, registry and issue closure. | Final receipt matrix with authority and state transition for each lane. | Archive only repository-complete work; leave external lanes pending. |

## Options and rationale

### Option A — Recommended: hosted exact-head closeout

Push the validated branch, run protected checks, refresh the installed parity
packet on the hosted runner, then reconcile receipts. This provides the
strongest evidence and preserves exact-head policy. It depends on GitHub runner
and account access.

### Option B — Local-only repository closeout

Complete every local gate and archive with external gates explicitly pending.
This is appropriate if hosted access is unavailable, but it cannot close G10
promotion, G13 live reconciliation, or G14 hosted acceptance.

### Option C — Defer external lanes

Freeze the current evidence packet, mark the remaining lanes blocked, and open
follow-up tasks for hosted parity, scientific approval and registry receipts.
This minimizes risk but leaves the programme intentionally incomplete.

## Current recommendation and contingencies

Proceed with Option A when Git transport and hosted runners are available. Use
Option B only for repository documentation/validation completion, and Option C
if hosted or accountable human authorities remain unavailable after the local
packet is complete. In every option, scientific review, signing, publication,
registry acceptance and parent-issue closure remain separate decisions.
