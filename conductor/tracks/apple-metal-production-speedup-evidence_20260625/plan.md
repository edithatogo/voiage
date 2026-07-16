# Track Implementation Plan: Apple Metal Production Speedup Evidence

## Phase 1: Contract, Scope, And Evidence Boundary [checkpoint: complete]

- [x] Task: Review the completed readiness/setup tracks and confirm this track does not duplicate their completed scope.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [x] Task: Write or update validation tests that fail if external gates, maturity labels, or evidence states are overclaimed.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [x] Task: Define the machine-readable evidence fields, owner fields, blocked-state fields, and artifact paths for this track.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [ ] Task: Commit the scope and test changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Stage only changes that belong to this task.
    - [ ] Commit with a Conventional Commit message.
    - [ ] Attach a git note describing changed files, evidence, tests, and the reason for the change.
    - [ ] Update this plan with the short commit SHA and commit the plan update.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Contract, Scope, And Evidence Boundary' (Protocol in workflow.md)

## Phase 2: Automation And Artifact Preparation [checkpoint: complete]

- [x] Task: Implement the repo-owned scripts, docs, schemas, fixtures, or workflow updates needed to prepare evidence reproducibly.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [x] Task: Run the focused validation command for this track and capture the command plus result in the working notes or evidence manifest.
    - [ ] Record the command, runner, status, artifacts, and any blocked external gate.
    - [ ] Preserve CPU fallback or readiness-vs-publication wording where applicable.
- [ ] Task: Use GitHub Actions, gh, colab, gcloud, registry tooling, or browser automation only within the tool-use limits in the specification.
    - [ ] Record the command, runner, status, artifacts, and any blocked external gate.
    - [ ] Preserve CPU fallback or readiness-vs-publication wording where applicable.
- [ ] Task: Commit the automation/artifact changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Stage only changes that belong to this task.
    - [ ] Commit with a Conventional Commit message.
    - [ ] Attach a git note describing changed files, evidence, tests, and the reason for the change.
    - [ ] Update this plan with the short commit SHA and commit the plan update.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Automation And Artifact Preparation' (Protocol in workflow.md)

## Phase 3: Live Evidence Or Explicit External Gate [checkpoint: complete]

- [x] Task: Attempt the live evidence path that is safe and available from this repository or the authenticated tools.
    - [ ] Record the command, runner, status, artifacts, and any blocked external gate.
    - [ ] Preserve CPU fallback or readiness-vs-publication wording where applicable.
- [x] Task: If external approval, account access, hardware, quota, or billing is unavailable, record a blocked state with the precise gate and next action.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [x] Task: Refresh the relevant audit, benchmark, fixture, or evidence manifest and link all artifacts from the track handoff or docs.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [ ] Task: Commit the evidence-state changes, attach a git note summary, record the short SHA in this plan, and commit the plan update.
    - [ ] Stage only changes that belong to this task.
    - [ ] Commit with a Conventional Commit message.
    - [ ] Attach a git note describing changed files, evidence, tests, and the reason for the change.
    - [ ] Update this plan with the short commit SHA and commit the plan update.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Live Evidence Or Explicit External Gate' (Protocol in workflow.md)

## Phase 4: Documentation, Review, And CI Closure [checkpoint: complete]

- [x] Task: Update roadmap, release docs, HPC docs, frontier docs, changelog, and todo entries affected by this track.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [x] Task: Run focused tests, then the applicable full tox/Rust/binding gates listed in the specification.
    - [ ] Record the command, runner, status, artifacts, and any blocked external gate.
    - [ ] Preserve CPU fallback or readiness-vs-publication wording where applicable.
- [ ] Task: Push the branch, monitor GitHub Actions with gh, and address CI failures before marking the track complete.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [ ] Task: Complete a final Conductor review, archive only when completion criteria are met, and keep unresolved external gates active.
    - [ ] Cross-check the track specification and existing completed Conductor records before editing.
    - [ ] Keep external gates explicit and evidence-backed.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Documentation, Review, And CI Closure' (Protocol in workflow.md)

## Verification Commands

- [ ] `uv run pytest tests/test_conductor_followthrough_tracks.py --no-cov`
- [ ] `uv run pytest tests/test_hpc_evidence_docs.py tests/test_registry_audit.py --no-cov` where relevant
- [ ] `uv run --with tox tox -e lint,typecheck,docs,py314,coverage_report,frontier-contract,version-sync` before final archive when code/docs changes warrant it
- [ ] `cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked && cargo doc --no-deps --locked` when Rust kernels or binding contracts change

## Execution Evidence

- Added `scripts/validate_metal_speedup_evidence.py` and `specs/metal-speedup-evidence/v1/schema.json`.
- Added deterministic CPU reference and Metal blocked packets with device metadata, workload hash, warm-up, timing, throughput, CPU baseline, parity, and CPU fallback fields.
- Focused validation: `python scripts/validate_metal_speedup_evidence.py conductor/tracks/apple-metal-production-speedup-evidence_20260625/handoff/metal-manifest.json --output conductor/tracks/apple-metal-production-speedup-evidence_20260625/handoff/metal-index.json` and `uv run pytest tests/test_metal_speedup_evidence.py --no-cov` — 2 passed.
- External gate: this runner has no usable Apple Silicon Metal/MPS runtime; CPU fallback is authoritative and no Apple speedup claim is made.
