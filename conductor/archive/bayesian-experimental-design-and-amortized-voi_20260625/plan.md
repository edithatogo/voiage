# Track Implementation Plan: Bayesian Experimental Design And Amortized VOI

## Completion record

- [x] Implemented backend-neutral expected information gain estimation,
  cost-aware Bayesian design selection, active-learning batch selection, and
  amortized EVSI summaries in ``voiage/experimental_design.py``.
- [x] Added focused tests and experimental-boundary documentation.
- [x] Local gates passed: lint, Bandit, typecheck, Astro check/build, 1213
  Python 3.14 tests, 90.88% coverage, frontier contract, and version sync.
- [x] GitHub PR #155 merged with all required checks passing; merge commit is
  ``eda140a``. The repository ruleset was restored active after the approved
  administrative merge.
- [x] Archived after review. Optional JAX/NumPyro/SBI backends, real-data
  evidence, and cross-language parity remain explicit downstream gates.

## Phase 1: Architecture And Gate Definition [checkpoint: ]

- [x] Task: Review existing roadmap, Conductor records, dependencies, and fixtures before implementation changes.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Add or update validation tests for the new policy, architecture, or method boundary.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit the tests/policy boundary, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Architecture And Gate Definition' (Protocol in workflow.md)

## Phase 2: Implementation Or Evidence Artifact [checkpoint: ]

- [x] Task: Implement the docs, schemas, scripts, workflows, datasets, kernels, or examples defined by this track.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [x] Task: Run focused validation and record command, runner, status, and artifact paths.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit the implementation/evidence changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implementation Or Evidence Artifact' (Protocol in workflow.md)

## Phase 3: Integration And Cross-Track Review [checkpoint: ]

- [ ] Task: Verify this track does not conflict with registry, frontier, Rust, dataset, or HPC follow-through tracks.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Run applicable tox/Rust/binding/docs gates and record any blocked external or expensive gates.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Commit the integration/review changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Keep evidence, command output, blocked states, and external gates explicit.
    - [ ] Preserve commit notes, git notes, short commit SHA updates, and plan-update commits.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration And Cross-Track Review' (Protocol in workflow.md)

## Verification Commands

- [x] `uv run pytest tests/test_conductor_followthrough_tracks.py --no-cov`
- [x] `uv run --with tox tox -e lint,typecheck,docs,py314,coverage_report,frontier-contract,version-sync` — pass; 1213 passed, 14 skipped, 90.88% coverage
- [ ] Rust and binding language-native gates when kernels or adapters change

## Current checkpoint

- Implemented the backend-neutral ``voiage.experimental_design`` API for
  expected information gain, Bayesian design selection, active learning, and
  amortized EVSI summaries.
- Added deterministic unit coverage in ``tests/test_experimental_design.py``
  and documented the optional-backend boundary in the developer guide.
- Focused verification: 19 tests passed; repository harness passed with zero
  findings.
- Full local gate: tox passed with lint, Bandit, typecheck, Astro check/build,
  Python 3.14 tests, 90.88% coverage, frontier contract, and version sync.
- Implementation commit: ``bf4052a``; git note attached. PR #155 carries the
  hosted GitHub Actions validation. Heavy Bayesian backends, real-data
  evidence, and cross-language parity remain external/downstream gates.
