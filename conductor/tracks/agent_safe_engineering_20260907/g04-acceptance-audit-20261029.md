# G04 acceptance audit, 2026-10-29

Owner: conductor integrator, worktree
`/Users/doughnut/.codex/worktrees/ci-measurement-reconciliation/voiage`, branch
`codex/ci-measurement-reconciliation`. Scope: review and reconciliation of the
G04 acceptance claim; implementation and final acceptance remain open.

The prior merged receipt and packet are historical evidence of the bounded
change, not proof of the entire acceptance criterion. In particular:

* `ci-measurements.json` contains pending lock and coverage digests, no cold or
  warm durations, and no test count. It makes no measured savings claim.
* `test_ci_measurements_bind_reproducibility_inputs` checks field names and
  declared policy values. It does not run artifact identity comparisons, failed
  subprocesses, cancelled shards, missing coverage or retry handling.
* The preview nextest job runs a non-authoritative inventory and records cache
  statistics; it does not configure resource groups or demonstrate replay.

Before acceptance, qualify executable identity checks and fail-closed outcomes,
obtain comparable cold/warm measurements with exact test counts if claiming
savings, and evaluate resource groups and replay in the existing preview lane.
Keep the hosted required checks and thresholds unchanged. A future implementation
must freeze its own baseline and retain this correction alongside prior receipts.
