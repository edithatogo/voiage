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

The `ci-artifact-identity` lane binds the existing operational coverage report
to the clean source revision, lockfile, interpreter, coverage configuration,
declared features and artifact digest, and verifies the receipt before the
coverage gate consumes the report. Guarded CLI tests exercise wrong identity,
changed or missing bytes, dirty source and invalid receipts. The receipt is not
a signature, does not authenticate its producer, and currently covers only a
same-job artifact; it does not qualify cross-run reuse or shard aggregation.

Current implementation owner: this chat's integrator, worktree
`/Users/doughnut/.codex/worktrees/ci-artifact-identity/voiage`, scoped to the
coverage identity script, its tests, the operational assurance job and this
handoff. Native scope: none. Next operation: run final gates and hosted review.

The `ci-nextest-replay` lane adds a checkpoint-I/O group and a same-job archive
replay to the existing non-blocking preview. On macOS, pinned nextest 0.9.114
listed six `evpi_resume` tests in the one-thread group and replayed all six from
an archive under the Python 3.15.1 resource guard. The experiment does not
establish hosted Linux behavior, cross-run artifact identity, failure aggregation,
or a measured saving. Those acceptance requirements remain open.

Current implementation owner: this chat's integrator, worktree
`/Users/doughnut/.codex/worktrees/ci-nextest-replay/voiage`, scoped to
`rust/.config/nextest.toml`, the Rust preview job, its focused test and this
handoff. Native scope: none. Next operation: qualify the candidate with local
and hosted gates before merge.

The added `tests/test_coverage_cli_acceptance.py` invokes the existing coverage
gate in a subprocess: a missing report fails, an exact-head control passes, and
a mismatched checkout fails. This tests the current single-job coverage gate;
it does not establish identity checking for reused artifacts or aggregation of
cancelled shards, which remain open.

The same test file executes the current coverage-collection shell step with a
test runner that exits 37 and confirms that the step propagates that failure.
This is local command evidence, not a hosted cancellation or aggregate receipt.

An attempted aggregate in `dependency-preview.yml` removed the shard job's
`continue-on-error` and checked its matrix result. The focused run exposed an
existing contract in `test_dependency_promotion_policy.py` requiring every job
in that workflow to be non-blocking. That test is also a byte-pinned input to
the optional telemetry packet. The attempted workflow change was withdrawn;
the failure is evidence that a fail-closed aggregate needs a separately
qualified lane or coordinated revision of that contract and packet. No native
GitHub cancellation behavior was demonstrated by this local attempt.
