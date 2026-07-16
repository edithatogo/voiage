# Track Implementation Plan: Capacity And Budget-Constrained VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: completed]

- [x] Task: Audit existing implementation, portfolio, and frontier contract surfaces for overlap and compatibility.
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

## Phase 3: Cross-Language And Quality Gates [checkpoint: completed]

- [x] Task: Add cross-language conformance fixtures and adapter expectations for relevant bindings.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Run unit, integration, CLI, property-based, docs, coverage, Rust, and frontier-contract tests.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Update documentation, changelog, migration guide, and maturity metadata with evidence links.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit parity/quality changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Cross-Language And Quality Gates' (Protocol in workflow.md)

## Phase 4: Mature Stable Promotion Review [checkpoint: blocked_external]

- [ ] Task: Complete the frontier stable-promotion checklist and record the go/no-go decision.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: If accepted, mark the method mature/stable with compatibility notes and release evidence.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: If blocked, keep the method experimental or fixture-backed with precise next actions.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit the promotion decision, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Mature Stable Promotion Review' (Protocol in workflow.md)

## Verification Commands

- [x] `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90` — 1344 passed, 10 skipped, coverage 90.08%.
- [x] `uv run --with tox tox -e lint,typecheck,docs,frontier-contract` — passed.
- [x] `python scripts/validate_frontier_contract.py` — 17 families validated.
- [x] GitHub Actions and Astro documentation gates — local equivalents passed; hosted checks remain required on the PR.
- [ ] Rust and binding language-native gates when kernels or adapters change — deferred because this slice has no Rust kernel or adapter.

## Evidence checkpoint

- Implementation commit: `259f374` (short commit SHA), with a git note recording scope and evidence.
- Local artifacts: runtime in `voiage/methods/capacity_budget_constrained.py`, CLI command, deterministic fixtures under `specs/frontier/capacity-budget-constrained/v1`, tests, and Astro documentation.
- External gates: licensed constrained-allocation data, Rust/binding parity, and mature/stable approval remain blocked; no stable claim is made.

## Archive checkpoint

- Hosted implementation PR #185 merged as `38f4590`; all substantive GitHub Actions checks passed.
- The completed track is archived after local and hosted gates; the executable queue remains orders 18 through 32.
