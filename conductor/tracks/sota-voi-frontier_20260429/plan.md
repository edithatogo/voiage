# Track Implementation Plan: SOTA VOI Frontier

## Phase 1: Frontier Method Audit And Contract Scope [checkpoint: ]

- [x] Task: Create the frontier VOI research note.
    - [x] Summarize the method families not yet represented in the stable API.
    - [x] Separate already implemented methods from planned frontier methods.
    - [x] Identify which methods are health-economics specific and which are domain-agnostic.
- [x] Task: Define method maturity labels.
    - [x] Add categories for planned, experimental, fixture-backed, and stable.
    - [x] Require each new method to expose diagnostics and maturity metadata.
- [x] Task: Triage adjacent frontier extensions.
    - [x] Decide whether causal-identification, transportability, and external-validity VOI belong in the core API or extension tracks.
    - [x] Decide whether data-quality, measurement-error, data-acquisition, privacy, and linkage VOI need separate schemas.
    - [x] Decide whether computational VOI, model-refinement VOI, expert-elicitation VOI, and evidence-synthesis design VOI should become implementation tracks.
- [x] Task: Conductor - Automated Review and Checkpoint 'Frontier Method Audit And Contract Scope' (Protocol in workflow.md)

## Phase 2: Value Of Perspective Contract [checkpoint: ]

- [x] Task: Define the multi-perspective input contract.
    - [x] Specify perspective identifiers, labels, cost categories, effects, utility weights, equity weights, WTP thresholds, discount rates, and population scaling.
    - [x] Specify the sample x strategy x perspective net-benefit shape.
    - [x] Specify validation behavior for misaligned strategies, samples, and perspectives.
- [x] Task: Define Value of Perspective result contracts.
    - [x] Include perspective-specific decisions and expected net benefits.
    - [x] Include cross-perspective regret and switching-value matrices.
    - [x] Include robust/consensus strategy outputs under perspective weights.
    - [x] Include Pareto/non-dominated strategy results across perspectives.
- [x] Task: Add v1 schemas and examples for perspective inputs and outputs.
- [x] Task: Conductor - Automated Review and Checkpoint 'Value Of Perspective Contract' (Protocol in workflow.md)

## Phase 3: Value Of Perspective Implementation [checkpoint: ]

- [x] Task: Add tests for perspective comparison behavior.
    - [x] Test identical perspectives produce zero switching value.
    - [x] Test conflicting perspectives produce nonzero regret.
    - [x] Test weighted consensus strategy selection.
    - [x] Test Pareto/non-dominated strategy extraction.
- [x] Task: Implement the public method surface.
    - [x] Add a method module for perspective analysis.
    - [x] Add high-level `DecisionAnalysis` wrappers.
    - [x] Add curated package exports.
- [x] Task: Add CLI and plotting support.
    - [x] Add a CLI command for multi-perspective comparison.
    - [x] Add compact tabular output plus JSON output.
    - [x] Add a perspective comparison plot.
- [x] Task: Add deterministic conformance fixtures for Value of Perspective.
    - [x] Add the screening-program fixture set and CLI payload expectation.
- [x] Task: Conductor - Automated Review and Checkpoint 'Value Of Perspective Implementation' (Protocol in workflow.md)

## Phase 4: Distributional, Equity, And Implementation-Adjusted VOI [checkpoint: ]

- [x] Task: Define distributional and equity-weighted VOI contracts.
    - [x] Support subgroup distributions, equity weights, and social-welfare summaries.
    - [x] Clarify relationship to existing Value of Heterogeneity methods.
    - Experimental Python API, `DecisionAnalysis` wrapper, exports, and tests are in place.
- [x] Task: Define implementation-adjusted VOI contracts.
    - [x] Support uptake, adherence, coverage, implementation delay, and implementation uncertainty.
    - [x] Coordinate artifact expectations with the ecosystem `innovate` track.
    - Experimental Python API, `DecisionAnalysis` wrapper, exports, and tests are in place.
- [x] Task: Add first deterministic fixtures for equity and implementation-adjusted VOI.
    - [x] Add versioned experimental schema and example payloads for distributional/equity VOI.
    - [x] Add versioned experimental schema and example payloads for implementation-adjusted VOI.
    - [x] Add normative fixture manifests and deterministic input/output payloads for both methods.
- [x] Task: Conductor - Automated Review and Checkpoint 'Distributional, Equity, And Implementation-Adjusted VOI' (Protocol in workflow.md)

## Phase 5: Preference, Validation, Threshold, And Robust VOI [checkpoint: b232d31]

- [x] Task: Define preference heterogeneity and value of individualized care contracts.
    - [ ] Model preference weights separately from clinical or cost uncertainty.
    - [ ] Support individual, subgroup, and population-level summaries.
- [x] Task: Define model-validation VOI contracts.
    - [x] Cover external validation of prediction models and model discrepancy reduction.
- [x] Task: Define threshold, tipping-point, and robust VOI contracts.
    - [x] Cover threshold-crossing probability, decision reversals, ambiguity sets, and robust strategy value.
- [x] Task: Conductor - Automated Review and Checkpoint 'Preference, Validation, Threshold, And Robust VOI' (Protocol in workflow.md)

## Phase 6: Documentation, Roadmap, And Release Readiness [checkpoint: ]

- [x] Task: Update README, migration guide, roadmap, and user-guide docs.
- [~] Task: Add CHEERS-VOI reporting metadata and structured result fields.
    - [x] Add shared CHEERS-VOI reporting payloads to the experimental Value of Perspective, distributional/equity, and implementation-adjusted result objects.
    - [ ] Expand the reporting payloads to the remaining frontier families and make them fixture-backed.
- [x] Task: Add examples for perspective comparison and at least one additional frontier method.
- [x] Task: Add release notes and experimental API warnings.
- [x] Task: Run full verification with `tox -e lint,typecheck,coverage_report` and the full pytest coverage suite.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Documentation, Roadmap, And Release Readiness' (Protocol in workflow.md)

## Phase 7: Dynamic Real-Options VOI [checkpoint: ]

- [x] Task: Define dynamic real-options VOI contracts.
    - [x] Model delay, irreversibility, and policy lock-in as explicit terms.
    - [x] Specify staged evidence arrival, action timing, and option exercise rules.
    - [x] Require diagnostics, maturity metadata, and reproducibility fields.
- [ ] Task: Add schemas and examples for dynamic real-options VOI inputs and outputs.
- [ ] Task: Add deterministic fixtures and reviewable example payloads once the contract stabilizes.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options VOI' (Protocol in workflow.md)

## Phase 8: Adjacent Frontier Extensions [checkpoint: ]

- [ ] Task: Define causal-identification, transportability, and external-validity VOI contracts.
    - [ ] Model source-to-target population shifts, transport weights, and validity penalties.
    - [ ] Distinguish internal validity, external validity, and transportability assumptions.
- [ ] Task: Define data-quality, measurement-error, data-acquisition, privacy, and linkage VOI contracts.
    - [ ] Model operational acquisition costs and privacy-constrained information value.
    - [ ] Distinguish source quality, linkage quality, and missingness/measurement-error value.
- [ ] Task: Define computational VOI, model-refinement VOI, expert-elicitation VOI, and evidence-synthesis design VOI contracts.
    - [ ] Model compute budget, approximation error, and refinement value explicitly.
    - [ ] Distinguish elicitation design value from downstream decision value.
- [ ] Task: Add schemas, examples, fixtures, and maturity metadata for each adjacent extension family as it graduates from triage.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Adjacent Frontier Extensions' (Protocol in workflow.md)

## Execution Notes

- Implement Value of Perspective first; it is the clearest user-facing
  differentiator and can reuse net-benefit-first infrastructure.
- Keep planned frontier methods documented until their mathematical contracts,
  schemas, fixtures, and tests are ready.
- Treat dynamic real-options and the adjacent extension families as tracked
  backlog items rather than informal ideas.
- Do not add heavy new dependencies without a dependency-policy update.
