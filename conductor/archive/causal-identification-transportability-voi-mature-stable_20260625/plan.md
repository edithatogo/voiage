# Track Implementation Plan: Causal Identification Transportability And External Validity VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: ]

- [ ] Task: Audit existing contract scaffolds, docs, and runtime surfaces for this method family.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Define stable result envelopes, diagnostics, maturity labels, and external assumptions.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Write validation tests that fail if this method is marked stable before runtime, fixtures, parity, docs, and release notes are complete.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit the tests and boundary docs, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Contract And Maturity Boundary' (Protocol in workflow.md)

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: ]

- [ ] Task: Implement or extend Python runtime APIs, result objects, CLI commands, and deterministic synthetic fixtures.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Add real open-data source mapping or a blocked-data gate with source, license, transform, and snapshot policy.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Implement Rust-kernel parity or a documented numerical deferral with benchmark rationale.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit runtime/example changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Runtime, Fixtures, And Examples' (Protocol in workflow.md)

## Phase 3: Cross-Language And Quality Gates [checkpoint: ]

- [ ] Task: Add cross-language conformance fixtures and adapter expectations for relevant bindings.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Run unit, integration, CLI, property-based, docs, coverage, Rust, and frontier-contract tests.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Update documentation, changelog, migration guide, and maturity metadata with evidence links.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit parity/quality changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Cross-Language And Quality Gates' (Protocol in workflow.md)

## Phase 4: Mature Stable Promotion Review [checkpoint: ]

- [ ] Task: Complete the frontier stable-promotion checklist and record the go/no-go decision.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: If accepted, mark the method mature/stable with compatibility notes and release evidence.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: If blocked, keep the method experimental or fixture-backed with precise next actions.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit the promotion decision, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Preserve evidence links, commands, artifact paths, blocked gates, and maturity status.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Mature Stable Promotion Review' (Protocol in workflow.md)

## Verification Commands

- [ ] `uv run pytest tests/test_conductor_followthrough_tracks.py --no-cov`
- [ ] `uv run --with tox tox -e lint,typecheck,docs,py314,coverage_report,frontier-contract,version-sync`
- [ ] Rust and binding language-native gates when kernels or adapters change

## Evidence update 2026-07-17

- Implemented the fixture-backed Python runtime and result envelope in
  ``voiage/methods/causal_transportability.py`` and exposed it through the
  curated package API and ``calculate-causal-transportability`` CLI command.
- Updated the v1 schemas and examples to ``fixture-backed`` maturity and added
  hash-pinned evidence in ``specs/frontier/causal-transportability/v1/fixtures/evidence.json``.
- Added runtime, contract, CLI, export, and registry tests. Local focused
  validation passed; hosted PR #167 passed all required jobs and Python CodeQL.
- The method remains fixture-backed. Licensed open-data attribution,
  cross-language/Rust parity, and mature/stable approval remain explicit gates.

## Archive closeout 2026-07-17

- Hosted PR #167 merged as ``6d1caeb`` with required checks and Python CodeQL
  passing; the maximal-quality ruleset was restored active.
- Archived after completing the repository-owned implementation and evidence
  slice. External gates remain recorded above and in the evidence manifest.
