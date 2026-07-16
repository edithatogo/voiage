# Track Implementation Plan: Adaptive Learning And Bandit VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: a88189d]

- [x] Audit existing adaptive-trial, sequential, CLI, frontier, and Astro surfaces.
- [x] Define the fixture-backed result envelope, diagnostics, policy inputs, stopping controls, and external assumptions.
- [x] Add contract tests that prevent stable promotion before runtime, fixtures, parity, docs, and release evidence are complete.
- [x] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
- [x] Conductor - User Manual Verification: Phase 1 (Protocol in workflow.md).

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: a88189d]

- [x] Implement dependency-free UCB, Thompson, and epsilon-greedy sequential allocation with exploration cost, regret, switching, burden, and stopping diagnostics.
- [x] Add `DecisionAnalysis.value_of_adaptive_learning_bandit`, the `calculate-adaptive-learning-bandit` CLI command, deterministic normative fixtures, and schema.
- [x] Record the open-data blocker: no licensed online allocation trace is committed.
- [x] Record the cross-language/Rust parity deferral and keep method maturity `fixture-backed`.
- [x] Commit runtime/example changes as `a88189d` with a git note, record the short commit SHA, and commit the plan update separately.
- [x] Conductor - User Manual Verification: Phase 2 (Protocol in workflow.md).

## Phase 3: Cross-Language And Quality Gates [checkpoint: a88189d]

- [ ] Add cross-language conformance fixtures once binding adapters are available.
- [x] Run unit, integration, CLI, property-based, docs, coverage, Rust, and frontier-contract tests: 1,335 passed, 10 optional skips, coverage 90.03%.
- [x] Update changelog, Astro migration guide, migration note, frontier registry, governance checklist, and maturity metadata.
- [ ] Commit parity/quality changes with a git note, a short commit SHA, and commit the plan update after external parity evidence exists.
- [ ] Conductor - User Manual Verification: Phase 3 (Protocol in workflow.md).

## Phase 4: Mature Stable Promotion Review [checkpoint: blocked]

- [ ] Complete the stable-promotion checklist.
- [ ] Obtain licensed online-allocation data and reproducible source/transform attribution.
- [ ] Complete cross-language/Rust parity and binding-native gates.
- [ ] Keep the method fixture-backed until those gates pass; do not claim stable promotion.
- [ ] Conductor - User Manual Verification: Phase 4 (Protocol in workflow.md).

## Verification Commands

- `uv run pytest tests/test_adaptive_learning_bandit.py tests/test_adaptive_learning_bandit_contract.py tests/test_adaptive_learning_bandit_cli.py --no-cov`
- `uv run --with tox tox -e lint,typecheck,docs,frontier-contract`
- `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90`
- `python scripts/validate_frontier_contract.py`
- Hosted GitHub Actions must pass the repository harness, quality, coverage, frontier, version, dependency, and CodeQL checks before merge.

## Evidence And Handoff

- Implementation commit: `a88189d` (`feat: add adaptive learning bandit voi`); git note attached.
- Plan update commit: `f94c173`; git note attached.
- Hosted merge gate: pending GitHub Actions; do not merge while substantive checks or Python CodeQL are failing.
