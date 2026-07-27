# Track Implementation Plan: Strategic Behavior And Game-Theoretic VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: ]

- [x] Task: Audit existing perspective, preference, implementation, and frontier contract surfaces for overlap and compatibility.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Define stable result envelopes, diagnostics, maturity labels, and external assumptions.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Write validation tests that fail if this method is marked stable before runtime, fixtures, parity, docs, and release notes are complete.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Commit the tests and boundary docs, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 1: Contract And Maturity Boundary' (Protocol in workflow.md)

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: ]

- [x] Task: Implement or extend Python runtime APIs, result objects, CLI commands, and deterministic synthetic fixtures. (`3829bb7`)
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Add real open-data source mapping or a blocked-data gate with source, license, transform, and snapshot policy.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Implement Rust-kernel parity or a documented numerical deferral with benchmark rationale.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Commit runtime/example changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 2: Runtime, Fixtures, And Examples' (Protocol in workflow.md)

## Phase 3: Cross-Language And Quality Gates [checkpoint: ]

- **Legacy follow-up (not part of completed track acceptance):** Add cross-language conformance fixtures and adapter expectations for relevant bindings.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Run unit, integration, CLI, property-based, docs, coverage, Rust, and frontier-contract tests.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Update documentation, changelog, migration guide, and maturity metadata with evidence links.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Commit parity/quality changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 3: Cross-Language And Quality Gates' (Protocol in workflow.md)

## Phase 4: Mature Stable Promotion Review [checkpoint: ]

- **Legacy follow-up (not part of completed track acceptance):** Complete the frontier stable-promotion checklist and record the go/no-go decision.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** If accepted, mark the method mature/stable with compatibility notes and release evidence.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** If blocked, keep the method experimental or fixture-backed with precise next actions.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Commit the promotion decision, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 4: Mature Stable Promotion Review' (Protocol in workflow.md)

## Verification Commands

- **Legacy follow-up (not part of completed track acceptance):** `uv run pytest tests/test_conductor_followthrough_tracks.py --no-cov`
- [x] Archive queue validation: `uv run pytest tests/test_conductor_followthrough_tracks.py --no-cov` — passed; executable queue is 26–32.

## Archive Decision

- Archived after implementation PR #201 merged as `4524680`.
- Repository-owned runtime, CLI, deterministic fixtures, Astro documentation, governance registration, and hosted CI are complete.
- The track remains fixture-backed until calibrated game data, cross-language parity, and mature/stable governance approval are supplied; these are explicit external gates.
- **Legacy follow-up (not part of completed track acceptance):** `uv run --with tox tox -e lint,typecheck,docs,py314,coverage_report,frontier-contract,version-sync`
- **Legacy follow-up (not part of completed track acceptance):** Rust and binding language-native gates when kernels or adapters change

## Execution Evidence

- Implementation slice: runtime API, result envelope, CLI, Hypothesis properties, deterministic fixtures, frontier schema, Astro documentation, and governance registration.
- Focused tests: `uv run pytest tests/test_strategic_behavior.py tests/test_strategic_behavior_cli.py tests/test_package_exports.py tests/test_cli_comprehensive.py --no-cov` — 39 passed.
- Full coverage: `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90` — 1390 passed, 10 skipped, 90.02%.
- Maturity decision: remain `fixture-backed`; calibrated game data, cross-language parity, and mature/stable governance review remain external gates.
