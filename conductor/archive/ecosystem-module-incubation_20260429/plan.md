# Ecosystem Module Incubation Plan

## Phase 1: Role And Boundary Definition

- [x] Task: Document the current ecosystem roles.
    - [x] Define `voiage` as the VOI engine.
    - [x] Define `lifecourse` as the health-economic simulation producer.
    - [x] Define `innovate` as the diffusion/adoption scenario producer.
    - [x] Define `mars` as the fixed-API surrogate/metamodel backend candidate.
    - [x] Define HEOML as the shared portable artifact contract.
- [x] Task: Document non-goals.
    - [x] Exclude sibling project internals from stable integrations.
    - [x] Exclude pickle from portable interchange.
    - [x] Exclude direct modifications to `mars` core API.
- [x] Task: Conductor - Automated Review and Checkpoint 'Role And Boundary Definition' (Protocol in workflow.md)

## Phase 2: Artifact And Extension Contracts

- [x] Task: Define `voiage` ecosystem artifacts.
    - [x] Define consumed artifacts for net benefits, parameters, strategies, WTP thresholds, trial designs, and provenance.
    - [x] Define produced artifacts for EVPI, EVPPI, EVSI, ENBS, diagnostics, method settings, and plots.
    - [x] Align artifact metadata with HEOML extension naming.
- [x] Task: Define compatibility fixtures.
    - [x] Add fixture expectations for `lifecourse` to `voiage`.
    - [x] Add fixture expectations for `innovate` adoption uncertainty to VOI workflows.
    - [x] Add fixture expectations for optional `mars` metamodel backends.
- [x] Task: Conductor - Automated Review and Checkpoint 'Artifact And Extension Contracts' (Protocol in workflow.md)

## Phase 3: Dependency And Promotion Policy

- [x] Task: Define optional integration gates.
    - [x] Require optional extras for ecosystem adapters.
    - [x] Require smoke CI, Renovate coverage, security checks, docs, and removal paths.
    - [x] Require version compatibility matrices before supported status.
- [x] Task: Define promotion stages.
    - [x] Start with documented contract and fixtures.
    - [x] Promote to experimental adapter only after stable public APIs exist.
    - [x] Promote to supported adapter only after conformance and release policy are complete.
- [x] Task: Conductor - Automated Review and Checkpoint 'Dependency And Promotion Policy' (Protocol in workflow.md)

## Phase 4: Documentation And Roadmap Integration

- [x] Task: Update docs and specs.
    - [x] Add ecosystem module strategy documentation.
    - [x] Add the `specs/ecosystem/` contract outline.
    - [x] Link to the corresponding `lifecourse` ecosystem track and HEOML subtree.
- [x] Task: Update planning files.
    - [x] Update `roadmap.md`.
    - [x] Update `todo.md`.
    - [x] Update `changelog.md`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Documentation And Roadmap Integration' (Protocol in workflow.md)
