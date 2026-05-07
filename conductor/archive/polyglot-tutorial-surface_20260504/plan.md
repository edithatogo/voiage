# Track Implementation Plan: Polyglot Tutorial Surface And Worked Examples

## Phase 1: Inventory And Tutorial Policy [checkpoint: ]

- [x] Task: Audit the current tutorial and example surface across all languages.
  - [x] Inventory Python notebooks and docs examples.
  - [x] Inventory R package examples and long-form docs readiness.
  - [x] Inventory Go, Julia, Rust, TypeScript, and .NET walkthrough coverage.
- [x] Task: Define the tutorial policy and artifact types by language.
  - [x] Confirm which languages should use notebooks, vignettes, README walkthroughs, or sample programs.
  - [x] Define the canonical use cases every language must demonstrate.
  - [x] Define the smoke-test or verification rule for each tutorial type.
- [x] Task: Conductor - Automated Review and Checkpoint 'Inventory And Tutorial Policy' (Protocol in workflow.md)

## Phase 2: Python Tutorial Surface Consolidation [checkpoint: ]

- [x] Task: Normalize the Python notebook/tutorial index.
  - [x] Ensure the notebook index covers the main use cases and advanced methods.
  - [x] Make the notebook naming and descriptions consistent.
  - [x] Surface the notebook/tutorial entry points from the docs landing pages.
- [x] Task: Add or tighten validation for the Python tutorials.
  - [x] Execute the notebooks in CI-style mode.
  - [x] Verify the docs index links remain valid.
  - [x] Keep the tutorial examples synchronized with the stable Python API.
- [x] Task: Conductor - Automated Review and Checkpoint 'Python Tutorial Surface Consolidation' (Protocol in workflow.md)

## Phase 3: R Tutorial And Manual Integration [checkpoint: ]

- [x] Task: Create the R tutorial artifact.
  - [x] Add the vignette or equivalent long-form guide.
  - [x] Include the core EVPI, EVPPI, and EVSI workflows.
  - [x] Keep the tutorial aligned with the package-level docs track.
- [x] Task: Verify the R tutorial build path.
  - [x] Render the vignette or long-form guide in non-interactive mode.
  - [x] Ensure the manual and vignette outputs are non-empty and stable.
  - [x] Add or update tests that smoke-check the tutorial artifact.
- [x] Task: Conductor - Automated Review and Checkpoint 'R Tutorial And Manual Integration' (Protocol in workflow.md)

## Phase 4: Non-Python Binding Walkthroughs [checkpoint: ]

- [x] Task: Add a concise tutorial surface for Julia, Go, Rust, TypeScript, and .NET.
  - [x] Provide setup and first-analysis snippets in each binding README or dedicated guide.
  - [x] Keep each example runnable or at least dry-run friendly.
  - [x] Cover the core VOI use case in a language-idiomatic way.
- [x] Task: Add smoke checks for the binding walkthroughs.
  - [x] Validate the README snippets or sample programs where practical.
  - [x] Keep the examples aligned with the release workflows and package metadata.
  - The repository now includes a static smoke layer for the binding walkthrough READMEs, and the language-native build checks were exercised during the track review.
- [x] Task: Conductor - Automated Review and Checkpoint 'Non-Python Binding Walkthroughs' (Protocol in workflow.md)

## Phase 5: Docs Integration, Verification, And Handoff [checkpoint: ]

- [x] Task: Wire the tutorial surface into the main docs and release guidance.
  - [x] Link the tutorials from the top-level docs and language binding pages.
  - [x] Document how users should choose between notebooks, vignettes, and README examples.
  - [x] Record any language-specific limitations or caveats.
    - Python keeps the notebook-first surface; R uses a vignette plus PDF manual; the other binding walkthroughs currently live in READMEs.
- [x] Task: Run the full tutorial verification suite.
  - [x] Execute the notebook and tutorial checks.
  - [x] Run the relevant language build checks.
    - Python notebook execution, the R docs/manual build path, and the binding walkthrough smoke/build pass are verified; the .NET walkthrough notes the `net11.0` SDK requirement for local builds.
  - [x] Confirm the docs links and tutorial indexes are current.
- [x] Task: Conductor - Automated Review and Checkpoint 'Docs Integration, Verification, And Handoff' (Protocol in workflow.md)

## Execution Notes

- This track is intentionally split so each language family can be handled by different subagents.
- Keep the artifacts compact, reproducible, and user-facing.
- Prefer a small number of strong examples over a large number of thin ones.
