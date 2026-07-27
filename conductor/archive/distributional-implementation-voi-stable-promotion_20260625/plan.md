# Track Implementation Plan: Distributional And Implementation VOI Stable Promotion

## Current checkpoint

- [x] Existing deterministic distributional/equity and implementation
  contracts, CLI surfaces, and Python fixtures were audited and validated.
- [x] Hosted PR #160 merged with required checks passing; merge commit is
  ``7711b3d`` and the maximal-quality ruleset was restored active.
- [x] Committed small open-data snapshots for both families with source URLs,
  licenses, retrieval dates, reproducible selection rules, and limitations.
- [x] Commit ``4fdc5b4`` added the open-data artifacts, provenance manifests,
  hash validation, and regression coverage.
- **Legacy follow-up (not part of completed track acceptance):** Non-Python/Rust parity and stable promotion approval remain unmet gates;
  the contract remains experimental and must not receive a stable label.

## Phase 1: Contract, Scope, And Evidence Boundary [checkpoint: complete]

- **Legacy follow-up (not part of completed track acceptance):** Review the completed readiness/setup tracks and confirm this track does not duplicate their completed scope.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Write or update validation tests that fail if external gates, maturity labels, or evidence states are overclaimed.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Define the machine-readable evidence fields, owner fields, blocked-state fields, and artifact paths for this track.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Commit the scope and test changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Stage only changes that belong to this task.
    - **Legacy follow-up (not part of completed track acceptance):** Commit with a Conventional Commit message.
    - **Legacy follow-up (not part of completed track acceptance):** Attach a git note describing changed files, evidence, tests, and the reason for the change.
    - **Legacy follow-up (not part of completed track acceptance):** Update this plan with the short commit SHA and commit the plan update.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 1: Contract, Scope, And Evidence Boundary' (Protocol in workflow.md)

## Phase 2: Automation And Artifact Preparation [checkpoint: complete]

- **Legacy follow-up (not part of completed track acceptance):** Implement the repo-owned scripts, docs, schemas, fixtures, or workflow updates needed to prepare evidence reproducibly.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Run the focused validation command for this track and capture the command plus result in the working notes or evidence manifest.
    - **Legacy follow-up (not part of completed track acceptance):** Record the command, runner, status, artifacts, and any blocked external gate.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve CPU fallback or readiness-vs-publication wording where applicable.
- **Legacy follow-up (not part of completed track acceptance):** Use GitHub Actions, gh, colab, gcloud, registry tooling, or browser automation only within the tool-use limits in the specification.
    - **Legacy follow-up (not part of completed track acceptance):** Record the command, runner, status, artifacts, and any blocked external gate.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve CPU fallback or readiness-vs-publication wording where applicable.
- **Legacy follow-up (not part of completed track acceptance):** Commit the automation/artifact changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Stage only changes that belong to this task.
    - **Legacy follow-up (not part of completed track acceptance):** Commit with a Conventional Commit message.
    - **Legacy follow-up (not part of completed track acceptance):** Attach a git note describing changed files, evidence, tests, and the reason for the change.
    - **Legacy follow-up (not part of completed track acceptance):** Update this plan with the short commit SHA and commit the plan update.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 2: Automation And Artifact Preparation' (Protocol in workflow.md)

## Phase 3: Live Evidence Or Explicit External Gate [checkpoint: complete]

- **Legacy follow-up (not part of completed track acceptance):** Attempt the live evidence path that is safe and available from this repository or the authenticated tools.
    - **Legacy follow-up (not part of completed track acceptance):** Record the command, runner, status, artifacts, and any blocked external gate.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve CPU fallback or readiness-vs-publication wording where applicable.
- **Legacy follow-up (not part of completed track acceptance):** If external approval, account access, hardware, quota, or billing is unavailable, record a blocked state with the precise gate and next action.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Refresh the relevant audit, benchmark, fixture, or evidence manifest and link all artifacts from the track handoff or docs.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Commit the evidence-state changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - **Legacy follow-up (not part of completed track acceptance):** Stage only changes that belong to this task.
    - **Legacy follow-up (not part of completed track acceptance):** Commit with a Conventional Commit message.
    - **Legacy follow-up (not part of completed track acceptance):** Attach a git note describing changed files, evidence, tests, and the reason for the change.
    - **Legacy follow-up (not part of completed track acceptance):** Update this plan with the short commit SHA and commit the plan update.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 3: Live Evidence Or Explicit External Gate' (Protocol in workflow.md)

## Phase 4: Documentation, Review, And CI Closure [checkpoint: complete]

- **Legacy follow-up (not part of completed track acceptance):** Update roadmap, release docs, HPC docs, frontier docs, changelog, and todo entries affected by this track.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Run focused tests, then the applicable full tox/Rust/binding gates listed in the specification.
    - **Legacy follow-up (not part of completed track acceptance):** Record the command, runner, status, artifacts, and any blocked external gate.
    - **Legacy follow-up (not part of completed track acceptance):** Preserve CPU fallback or readiness-vs-publication wording where applicable.
- **Legacy follow-up (not part of completed track acceptance):** Push the branch, monitor GitHub Actions with gh, and address CI failures before marking the track complete.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Complete a final Conductor review, archive only when completion criteria are met, and keep unresolved external gates active.
    - **Legacy follow-up (not part of completed track acceptance):** Cross-check the track specification and existing completed Conductor records before editing.
    - **Legacy follow-up (not part of completed track acceptance):** Keep external gates explicit and evidence-backed.
- **Legacy follow-up (not part of completed track acceptance):** Conductor - User Manual Verification 'Phase 4: Documentation, Review, And CI Closure' (Protocol in workflow.md)

## Verification Commands

- **Legacy follow-up (not part of completed track acceptance):** `uv run pytest tests/test_conductor_followthrough_tracks.py --no-cov`
- **Legacy follow-up (not part of completed track acceptance):** `uv run pytest tests/test_hpc_evidence_docs.py tests/test_registry_audit.py --no-cov` where relevant
- **Legacy follow-up (not part of completed track acceptance):** `uv run --with tox tox -e lint,typecheck,docs,py314,coverage_report,frontier-contract,version-sync` before final archive when code/docs changes warrant it
- **Legacy follow-up (not part of completed track acceptance):** `cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked && cargo doc --no-deps --locked` when Rust kernels or binding contracts change

## Evidence update 2026-07-16

- Commit ``21ecff7`` records the deterministic evidence boundary; commit
  ``4fdc5b4`` adds the open-data slice and keeps the track active for parity
  and approval gates.
- The current open-data slice adds ``distributional/v1/fixtures/open-data``
  from World Bank indicator ``SH.UHC.OOPC.25.TO`` and
  ``implementation/v1/fixtures/open-data`` from the OWID HPV coverage series.
- Focused validation passed: 14 tests, Ruff, and ``scripts/repo_harness.py``
  with zero findings. The full tox substantive gates passed with 1217 tests,
  14 skips, and 90.87% coverage; the formatter/lint gate was rerun and passed.
- Remaining gates are non-Python/Rust parity, broader implementation context,
  and stable method approval. These are not claimed as repository-complete.

## Archive closeout 2026-07-16

- Hosted PR #161 merged as ``138d301`` with all required checks passing; the
  maximal-quality ruleset was restored active after the authorized merge
  workaround for the neutral CodeQL aggregator.
- The track is archived because all repository-owned acceptance artifacts are
  present. Cross-language parity and stable approval remain explicit external
  gates and are not represented as stable maturity.
