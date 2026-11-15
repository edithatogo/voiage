# Specification

## Problem and scope

The maintainer requested faster GitHub Actions on 15 November 2026. Recent
successful runs show both long runner queues and unnecessary dependency work.
The C15 telemetry/oracle job compiled DuckDB for 894.597 seconds even though
its probe and four test modules do not use the experimental extra.

Reduce setup cost using the repository's existing tools. Keep every workflow
trigger, job name, test selection, runtime pin, coverage threshold, platform,
source-identity assertion, artifact control and security permission unchanged.
An explicit import preflight must keep JAX and Polars oracle checks mandatory
when removing the unrelated experimental extra.

## Changes

- Run the Bash/Vale prose check without a Python project environment.
- Run the standard-library repository harness with the pinned Python directly.
- Bootstrap version, ingestion and coverage tox jobs from the existing locked
  `verification` group. Tox still builds and validates the installed package in
  its own environments. Prevent implicit project resynchronization on invocation.
- Remove the unused experimental extra only from C15 telemetry/oracles.
- Cancel obsolete PR revisions in dependency review and shard qualification;
  retain independent groups for merge-group and manual runs.

## Acceptance

1. Existing affected workflow/security contracts and the repository harness pass.
2. The harness executes without site packages; tox bootstrap contains only its
   locked tooling and does not install the project into the outer environment.
3. All four C15 test modules and the collector probe pass without experimental
   packages; JAX/Polars imports succeed and no oracle cases disappear.
4. The final candidate receives the required local tox gate where capacity
   permits and current-revision hosted checks. Unexecuted gates remain pending.
5. Record queue time separately from execution. Do not claim a measured overall
   speedup until comparable successful hosted results are available.

## Boundaries and rollback

No dependency updates, new tools, runner upgrades, reduced tests, path-based
omissions, coverage consolidation, cache of test results, release changes or
hosted protection changes. Existing dependency/security PRs keep ownership.
Reverting this isolated workflow change restores the prior setup behavior;
there is no data migration or public API change.
