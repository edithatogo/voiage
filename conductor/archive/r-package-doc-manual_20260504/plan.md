# Track Implementation Plan: R Package Documentation Manual And Vignette

## Phase 1: Package Documentation Policy And Metadata Cleanup [checkpoint: ]

- [x] Task: Define the R documentation artifact policy.
  - [x] Decide whether the binding ships only a PDF reference manual or both a PDF manual and a narrative vignette.
  - [x] Decide the expected release artifact name and distribution path.
- [x] Task: Clean the R package metadata and roxygen surface.
  - [x] Remove stale documentation tags from the R source.
  - [x] Ensure the package-level doc topic is generated cleanly.
  - [x] Sync DESCRIPTION and NAMESPACE with the current R surface.
- [x] Task: Add targeted tests for the documentation contract.
  - [x] Check the package help topic exists.
  - [x] Check the public Rd set matches the exported API.
  - [x] Check the roxygen/doc generation path does not drift.
- [x] Task: Conductor - Automated Review and Checkpoint 'Package Documentation Policy And Metadata Cleanup' (Protocol in workflow.md)

## Phase 2: Narrative Documentation And Example Coverage [checkpoint: ]

- [x] Task: Create the long-form R documentation source.
  - [x] Add a getting-started vignette or equivalent long-form guide.
  - [x] Cover Python environment setup and the first EVPI/EVPPI/EVSI workflow.
  - [x] Keep the examples small enough to render non-interactively.
- [x] Task: Add rendering and smoke tests for the narrative docs.
  - [x] Render the vignette or long-form guide in CI-style mode.
  - [x] Verify the generated artifact is non-empty and stable.
  - [x] Ensure documentation examples remain in sync with the wrapper behavior.
- [x] Task: Conductor - Automated Review and Checkpoint 'Narrative Documentation And Example Coverage' (Protocol in workflow.md)

## Phase 3: PDF Manual Generation And Release Integration [checkpoint: ]

- [x] Task: Add the PDF manual build path.
  - [x] Choose the manual-generation command or helper.
  - [x] Make the build deterministic and non-interactive.
  - [x] Verify the resulting PDF artifact is produced and readable.
- [x] Task: Wire the manual into release and CI guidance.
  - [x] Update the R binding release workflow or release instructions so the manual is generated or attached as intended.
  - [x] Document the manual workflow in the release docs.
  - [x] Document the local verification steps in CONTRIBUTING.
- [x] Task: Conductor - Automated Review and Checkpoint 'PDF Manual Generation And Release Integration' (Protocol in workflow.md)

## Phase 4: Verification, Docs Sync, And Handoff [checkpoint: ]

- [x] Task: Run the R package documentation verification suite.
  - [x] Build the package.
  - [x] Check the package.
    - `R CMD check --no-manual --as-cran` completed with one NOTE only: missing `Authors@R` in `DESCRIPTION`.
  - [x] Validate the manual and narrative docs again after all changes.
- [x] Task: Sync the project-level documentation and backlog.
  - [x] Update roadmap notes for the R documentation/manual policy.
  - [x] Update todo entries to reflect the finished documentation track.
  - [x] Update the changelog with the user-facing docs improvement.
- [x] Task: Conductor - Automated Review and Checkpoint 'Verification, Docs Sync, And Handoff' (Protocol in workflow.md)

## Execution Notes

- Split the work so metadata cleanup, vignette content, manual generation, and docs/release updates can be handled by different subagents without overlapping writes.
- Keep the track focused on documentation quality and release readiness, not on changing runtime R behavior.
- The end state should satisfy a publication-quality R docs story even if the package is still distributed through GitHub Releases rather than CRAN.
