# Track Implementation Plan: AI-Assisted Evidence Triage VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: 4128707]

- [x] Task: Audit existing evidence-synthesis, expert-elicitation, and frontier contract surfaces for overlap and compatibility. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Define stable result envelopes, diagnostics, maturity labels, and external assumptions. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Write validation tests that fail if this method is marked stable before runtime, fixtures, parity, docs, and release notes are complete. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit the tests and boundary docs, attach a git note summary, record the short SHA in this plan, and commit the plan update. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Contract And Maturity Boundary' (Protocol in workflow.md)

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: 4128707]

- [x] Task: Implement or extend Python runtime APIs, result objects, CLI commands, and deterministic synthetic fixtures. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Add real open-data source mapping or a blocked-data gate with source, license, transform, and snapshot policy. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Implement Rust-kernel parity or a documented numerical deferral with benchmark rationale. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit runtime/example changes, attach a git note summary, record the short SHA in this plan, and commit the plan update. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Runtime, Fixtures, And Examples' (Protocol in workflow.md)

## Phase 3: Cross-Language And Quality Gates [checkpoint: 4128707]

- [x] Task: Add cross-language conformance fixtures and adapter expectations for relevant bindings. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Run unit, integration, CLI, property-based, docs, coverage, Rust, and frontier-contract tests. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Update documentation, changelog, migration guide, and maturity metadata with evidence links. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit parity/quality changes, attach a git note summary, record the short SHA in this plan, and commit the plan update. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Cross-Language And Quality Gates' (Protocol in workflow.md)

## Phase 4: Mature Stable Promotion Review [checkpoint: 4128707]

- [x] Task: Complete the frontier stable-promotion checklist and record the go/no-go decision. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: If accepted, mark the method mature/stable with compatibility notes and release evidence.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: If blocked, keep the method experimental or fixture-backed with precise next actions. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Commit the promotion decision, attach a git note summary, record the short SHA in this plan, and commit the plan update. (4128707)
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Mature Stable Promotion Review' (Protocol in workflow.md)

## Verification Commands

- [x] `uv run pytest tests/test_ai_assisted_evidence_triage.py tests/test_ai_assisted_evidence_triage_cli.py --no-cov` (6 passed, 2026-07-17)
- [x] `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90` (1360 passed, 10 skipped, 90.01%, 2026-07-17)
- [x] `uv run --with tox tox -e lint,harness,typecheck,docs,frontier-contract,version-sync` (passed, 2026-07-17)
- [x] `python scripts/validate_frontier_contract.py` (19 families validated, 2026-07-17)
- [ ] Rust and binding language-native gates when kernels or adapters change

## Review Evidence

- Automated Conductor review completed 2026-07-17; no unresolved correctness or contract blockers remained after the focused tests, full coverage, tox, and frontier validation.
- Implementation commit: `4128707`; git note records the runtime, CLI, fixture, documentation, tests, validation, and external gates.
- Promotion decision: no-go for mature/stable status. The method remains `fixture-backed` pending a licensed evidence corpus, external model validation, Rust/binding parity, and stable-promotion approval.
