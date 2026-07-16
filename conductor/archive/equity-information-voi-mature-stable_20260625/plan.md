# Track Implementation Plan: Equity Information VOI Mature Stable Path

## Phase 1: Contract And Maturity Boundary [checkpoint: 1027860]

- [x] Audit existing distributional/equity scaffolds and preserve the experimental-versus-fixture-backed boundary.
- [x] Define the result envelope, diagnostics, maturity label, parity deferral, and open-data gate.
- [x] Add validation tests preventing an unearned stable claim.
- [x] Commit `1027860` with git note evidence.
- [x] Conductor manual-verification boundary recorded as repository evidence; no external stable claim made.

## Phase 2: Runtime, Fixtures, And Examples [checkpoint: 1027860]

- [x] Implement `value_of_equity_information`, `EquityInformationResult`, `DecisionAnalysis` wrapper, and curated exports.
- [x] Add `calculate-equity-information`, a config template, deterministic normative fixtures, and migration documentation.
- [x] Record the licensed individual-level open-data gate as externally blocked with a precise next action.
- [x] Record Rust/binding parity as deferred because no adapter currently consumes this family.
- [x] Commit `1027860` with git note evidence.
- [x] Conductor manual-verification boundary recorded as repository evidence.

## Phase 3: Cross-Language And Quality Gates [checkpoint: 1027860]

- [x] Add JSON contract schema, fixture manifest, frontier registry entry, and promotion governance entry.
- [x] Run unit, integration, CLI, property-adjacent validation, docs, coverage, type, security, and frontier-contract gates.
- [x] Update Astro migration docs, changelog, and migration note with the promotion boundary.
- [x] Commit `1027860` with git note evidence.
- [x] Conductor manual-verification boundary recorded as repository evidence.

## Phase 4: Mature Stable Promotion Review [checkpoint: 1027860]

- [x] Record the promotion decision: keep the method `fixture-backed`; stable promotion is not accepted.
- [x] Record blocked external gates: licensed open-data attribution and cross-language/Rust parity.
- [x] Preserve precise next actions in `fixtures/evidence.json` and governance metadata.
- [x] Commit `1027860` with git note evidence.
- [x] Conductor manual-verification boundary recorded as repository evidence.

## Verification Evidence

- Conductor protocol records: short commit SHA `1027860`; commit the plan update;
  GitHub Actions evidence.
- Conductor - User Manual Verification: Phase 1 (Protocol in workflow.md).
- Conductor - User Manual Verification: Phase 2 (Protocol in workflow.md).
- Conductor - User Manual Verification: Phase 3 (Protocol in workflow.md).
- `uv run pytest --cov=voiage --cov-report=term --cov-fail-under=90`: 1301 passed, 10 skipped, 90.00%.
- `uv run --with tox tox -e lint,typecheck,docs,frontier-contract`: passed; Ruff, Bandit, ty, Astro check/build, and frontier validator all green.
- Hosted PR #179 merged as `74a3858`; all required substantive checks and Python CodeQL passed; aggregate CodeQL remained the known neutral wrapper.
