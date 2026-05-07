# Track Implementation Plan: Model Validation VOI Implementation

## Phase 1: Core Contract And Runtime Shape [checkpoint: ]

- [x] Task: Define the runtime validation-VOI module surface and result types.
    - [x] Choose the canonical method name and alias surface.
    - [x] Define the result envelope and reporting fields.
    - [x] Preserve compatibility with the validation contract scaffold.
- [x] Task: Add the public `DecisionAnalysis` wrapper and curated exports.
    - [x] Wire the method through `voiage/analysis.py`.
    - [x] Export the method and result types from `voiage/methods/__init__.py`.
    - [x] Export the method and result types from `voiage/__init__.py`.
- [x] Task: Add the initial export and input-validation tests.
    - [x] Lock the curated API surface in `tests/test_package_exports.py`.
    - [x] Cover invalid-shape and invalid-metadata branches.
- [x] Task: Conductor - Automated Review and Checkpoint 'Core Contract And Runtime Shape' (Protocol in workflow.md)

## Phase 2: Deterministic Validation Algorithm [checkpoint: ]

- [x] Task: Implement the validation-profile decision kernel.
    - [x] Compute profile-specific optima and expected net benefit.
    - [x] Compute discrepancy or regret summaries.
    - [x] Compute the value of external validation.
    - [x] Compute the value of model-discrepancy reduction.
- [x] Task: Implement robust and consensus summaries.
    - [x] Support profile weighting and reference-profile selection.
    - [x] Produce Pareto or non-dominated strategy sets.
- [x] Task: Add focused unit tests for the live validation algorithm.
    - [x] Cover the primary happy path.
    - [x] Cover profile-weight and reference-profile edge cases.
    - [x] Cover invalid input and shape errors.
- [x] Task: Conductor - Automated Review and Checkpoint 'Deterministic Validation Algorithm' (Protocol in workflow.md)

## Phase 3: Fixtures, CLI, And Documentation [checkpoint: ]

- [x] Task: Add deterministic fixtures and fixture-backed conformance tests.
    - [x] Create a validation fixture manifest and normative payload set.
    - [x] Add exact output comparisons against live runtime behavior.
    - [x] Wire the frontier registry and validator coverage to the new family.
- [x] Task: Add CLI integration if the method is user-facing from the command line.
    - [x] Add the command registration and argument parsing.
    - [x] Add JSON/CSV serialization branches where relevant.
    - [x] Add CLI regression tests.
- [x] Task: Update docs and roadmap references.
    - [x] Add an advanced VOI example or tutorial slice.
    - [x] Update the frontier docs and migration guide.
    - [x] Record the work in `roadmap.md`, `todo.md`, and `changelog.md`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Fixtures, CLI, And Documentation' (Protocol in workflow.md)

## Execution Notes

- Keep the validation track parallelizable: contract shape, algorithm, and
  fixtures/docs can be owned independently.
- Use deterministic payloads and exact comparisons to keep the frontier
  implementation reviewable.
