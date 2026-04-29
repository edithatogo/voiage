# Ecosystem Module Incubation Plan

## Phase 1: Role And Boundary Definition

- [ ] Task: Document the current HEOR ecosystem roles.
    - [ ] Define `voiage` as the HEOR VOI engine.
    - [ ] Define `lifecourse` as the health-economic simulation producer.
    - [ ] Define `innovate` as the health-intervention diffusion/adoption scenario producer.
    - [ ] Define `mars` as the fixed-API surrogate/metamodel backend candidate.
    - [ ] Define HEOML as the shared portable artifact contract.
- [ ] Task: Document non-goals.
    - [ ] Exclude generic non-HEOR modelling concerns.
    - [ ] Exclude sibling project internals from stable integrations.
    - [ ] Exclude pickle from portable interchange.
    - [ ] Exclude direct modifications to `mars` core API.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Role And Boundary Definition' (Protocol in workflow.md)

## Phase 2: Artifact And Extension Contracts

- [ ] Task: Define `voiage` HEOR ecosystem artifacts.
    - [ ] Define consumed artifacts for net benefits, parameters, strategies, WTP thresholds, trial designs, and provenance.
    - [ ] Define produced artifacts for EVPI, EVPPI, EVSI, ENBS, diagnostics, method settings, and plots.
    - [ ] Align artifact metadata with HEOML extension naming.
- [ ] Task: Define compatibility fixtures.
    - [ ] Add fixture expectations for `lifecourse` to `voiage`.
    - [ ] Add fixture expectations for `innovate` health-intervention adoption uncertainty to VOI workflows.
    - [ ] Add fixture expectations for optional `mars` metamodel backends.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Artifact And Extension Contracts' (Protocol in workflow.md)

## Phase 3: Dependency And Promotion Policy

- [ ] Task: Define optional integration gates.
    - [ ] Require optional extras for ecosystem adapters.
    - [ ] Require smoke CI, Renovate coverage, security checks, docs, and removal paths.
    - [ ] Require version compatibility matrices before supported status.
- [ ] Task: Define promotion stages.
    - [ ] Start with documented contract and fixtures.
    - [ ] Promote to experimental adapter only after stable public APIs exist.
    - [ ] Promote to supported adapter only after conformance and release policy are complete.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Dependency And Promotion Policy' (Protocol in workflow.md)

## Phase 4: Documentation And Roadmap Integration

- [ ] Task: Update docs and specs.
    - [ ] Add HEOR ecosystem module strategy documentation.
    - [ ] Add the `specs/ecosystem/` contract outline.
    - [ ] Link to the corresponding `lifecourse` ecosystem track and HEOML subtree.
- [ ] Task: Update planning files.
    - [ ] Update `roadmap.md`.
    - [ ] Update `todo.md`.
    - [ ] Update `changelog.md`.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Documentation And Roadmap Integration' (Protocol in workflow.md)
