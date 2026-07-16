# Track Implementation Plan: Ambiguity And Distribution Shift VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: c9a4a76]

- [x] Audit adjacent threshold, validation, frontier, CLI, and Astro surfaces.
- [x] Define the robust result envelope, diagnostics, fixture-backed maturity, and external assumptions.
- [x] Add runtime, contract, CLI, export, and regression tests for ambiguity, distribution shift, drift, robustness, and stable promotion boundaries.
- [x] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status in the track metadata.
- [x] Conductor - User Manual Verification: Phase 1 (Protocol in workflow.md).

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: c9a4a76]

- [x] Implement the Python result object, DecisionAnalysis wrapper, CLI command, deterministic normative fixture, and frontier schema.
- [x] Add migration documentation and an Astro user-guide entry; use Astro as the documentation platform.
- [x] Record the open-data blocker: no licensed source-target drift snapshot is committed.
- [x] Record the Rust/cross-language parity deferral and keep method maturity `fixture-backed`.
- [x] Implementation commit `c9a4a76` has a git note; this short commit SHA is recorded here, and this is the separate commit the plan update.
- [x] Conductor - User Manual Verification: Phase 2 (Protocol in workflow.md).

## Phase 3: Cross-Language And Quality Gates [checkpoint: c9a4a76]

- [x] Run focused runtime, contract, CLI, export, and frontier validation tests.
- [x] Run `git diff --check`, Ruff/Bandit, ty, Astro check/build, and frontier-contract validation.
- [x] Run the full suite with coverage: 1,317 passed, 10 optional-dependency skips, 90.01% coverage.
- [ ] Complete cross-language conformance once the external parity bindings and source snapshot are available.
- [x] Update changelog, migration guide, registry manifest, promotion checklist, and maturity metadata.
- [x] Conductor - User Manual Verification: Phase 3 (Protocol in workflow.md).

## Phase 4: Mature Stable Promotion Review [checkpoint: blocked]

- [x] Record the promotion decision: remain fixture-backed; stable promotion is not claimed.
- [ ] Obtain licensed open-data attribution and reproducible source-target drift snapshots.
- [ ] Complete cross-language parity and any binding-native gates.
- [ ] Revisit stable promotion only after those external gates pass; do not mark this track stable early.
- [ ] Conductor - User Manual Verification: Phase 4 (Protocol in workflow.md).

## Verification Commands

- `uv run pytest tests/test_ambiguity_distribution_shift.py tests/test_ambiguity_distribution_shift_contract.py tests/test_ambiguity_distribution_shift_cli.py --no-cov`
- `uv run --with tox tox -e lint,typecheck,docs,frontier-contract`
- `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90`
- `python scripts/validate_frontier_contract.py`
- Hosted GitHub Actions must pass the repository harness, quality, coverage, frontier, version, dependency, and CodeQL checks before merge.

## Evidence And Handoff

- Implementation commit: `c9a4a76` (`feat: add ambiguity and distribution shift voi`); git note attached.
- Plan update commit: pending; attach a git note and record its short commit SHA after this edit.
- Hosted merge gate: pending GitHub Actions; do not merge while substantive checks or Python CodeQL are failing.
- User-facing manual verification remains required under the Conductor protocol: verify CLI JSON output and the fixture-backed maturity boundary after hosted CI.
