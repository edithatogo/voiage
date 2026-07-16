# Track Implementation Plan: Federated And Privacy-Preserving VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: completed]

- [x] Task: Audit existing privacy, linkage, data-quality, and frontier contract surfaces for overlap and compatibility.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Define stable result envelopes, diagnostics, maturity labels, and external assumptions.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Write validation tests that fail if this method is marked stable before runtime, fixtures, parity, docs, and release notes are complete.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit the tests and boundary docs, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Contract And Maturity Boundary' (Protocol in workflow.md)

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: completed]

- [x] Task: Implement or extend Python runtime APIs, result objects, CLI commands, and deterministic synthetic fixtures.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Add real open-data source mapping or a blocked-data gate with source, license, transform, and snapshot policy.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Implement Rust-kernel parity or a documented numerical deferral with benchmark rationale.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit runtime/example changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Runtime, Fixtures, And Examples' (Protocol in workflow.md)

## Phase 3: Cross-Language And Quality Gates [checkpoint: 17700c2]

- [x] Task: Add cross-language conformance fixtures and adapter expectations for relevant bindings.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Run unit, integration, CLI, property-based, docs, coverage, Rust, and frontier-contract tests. (f76501a)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Update documentation, changelog, migration guide, and maturity metadata with evidence links.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit parity/quality changes, attach a git note summary, record the short SHA in this plan, and commit the plan update. (f76501a)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Cross-Language And Quality Gates' (Protocol in workflow.md)

## Phase 4: Mature Stable Promotion Review [checkpoint: pending archive checkpoint]

- [x] Task: Complete the frontier stable-promotion checklist and record the go/no-go decision.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: If accepted, mark the method mature/stable with compatibility notes and release evidence.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: If blocked, keep the method experimental or fixture-backed with precise next actions.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit the promotion decision, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Mature Stable Promotion Review' (Protocol in workflow.md)

## Verification Commands

- [x] `uv run pytest tests/test_federated_privacy_preserving.py tests/test_federated_privacy_preserving_cli.py --no-cov` (10 passed, 2026-07-17)
- [x] `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90` (1354 passed, 10 skipped, 90.03%, 2026-07-17)
- [x] `uv run --with tox tox -e lint,typecheck,docs,frontier-contract,version-sync` (passed, 2026-07-17)
- [x] `python scripts/validate_frontier_contract.py` (18 families validated, 2026-07-17)
- [ ] Rust and binding language-native gates when kernels or adapters change

## Review Evidence

- Automated Conductor review completed 2026-07-17; fixed the single-site leave-one-out division-by-zero edge case and added a regression test.
- Implementation commit: `f76501a`; git note attached with changed files, rationale, validation, and external gates.
- External gates remain blocked: licensed multi-site data acquisition, Rust/binding parity, and mature/stable promotion review. The method remains `fixture-backed`.
- Archive decision: repository-owned implementation and CI gates are complete; promotion is a no-go until the external gates above are evidenced.
