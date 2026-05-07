# Track Specification: SOTA VOI Frontier

## Overview

`voiage` should not stop at parity with BCEA, dampack, `voi`, or commercial
health-economic tools. This track defines the frontier method family needed for
`voiage` to become a state-of-the-art VOI library, including methods that are
not commonly available in existing packages.

The first new method family is Value of Perspective (VOP): the value and
decision consequences of assessing the same decision problem under multiple
perspectives, such as payer, societal, patient, provider, regulator, equity
weighted, or multi-stakeholder perspectives.

## Functional Requirements

1. Define a `PerspectiveSet` style contract for multiple decision perspectives.
   Each perspective must be able to carry cost categories, effect measures,
   utility weights, equity weights, willingness-to-pay thresholds, discount
   rates, population scaling, and stakeholder metadata.
2. Define a multi-perspective net-benefit representation with dimensions for
   sample, strategy, and perspective.
3. Implement Value of Perspective metrics:
   - perspective-specific optimal strategy and expected net benefit
   - cross-perspective regret matrix
   - value of switching perspective
   - robust or consensus strategy under weighted perspectives
   - Pareto/non-dominated strategy set across perspectives
4. Define plotting and CLI output requirements for comparing perspectives side
   by side.
5. Add frontier VOI method specifications and implementation tracks for:
   - distributional and equity-weighted VOI
   - implementation-adjusted VOI and value of implementation
   - preference heterogeneity and value of individualized care
   - value of model validation and prediction-model validation
   - threshold and tipping-point VOI
   - robust, ambiguity-aware, and multi-objective VOI
   - dynamic real-options style VOI where delay, irreversibility, or
     implementation timing affects decision value
   - causal-identification, transportability, and external-validity VOI
   - data-quality, measurement-error, data-acquisition, privacy, and linkage VOI
   - computational VOI and value of model refinement
   - expert-elicitation and evidence-synthesis design VOI
6. Keep these methods net-benefit-first and compatible with the core API
   contract and future bindings.
7. Add conformance fixtures for the new method families before exposing them as
   stable public APIs.
8. Add CHEERS-VOI-aligned reporting metadata, structured outputs, and
   reproducibility fields for each frontier method family before marking them
   stable.

## Non-Functional Requirements

1. The implementation must remain deterministic under seeded fixtures.
2. New methods must expose diagnostics and method maturity metadata.
3. New APIs must be documented as experimental until mathematical contracts,
   fixtures, CLI coverage, and cross-language schemas exist.
4. The base install must not gain heavy optional dependencies unless the
   dependency split is explicitly updated first.
5. The method names must avoid ambiguous overload of existing EVPI, EVPPI, EVSI,
   and ENBS semantics.

## Acceptance Criteria

1. A research note exists under `docs/` summarizing frontier method coverage and
   the rationale for the SOTA roadmap.
2. The roadmap and README feature matrix identify Value of Perspective and the
   broader frontier VOI method family.
3. A versioned schema/fixture plan exists for multi-perspective inputs and
   outputs.
4. At least the first implementation phase defines tests for perspective
   comparison before runtime code is added.
5. The track plan decomposes Value of Perspective separately from other frontier
   methods so it can be implemented first.
6. The reporting layer includes CHEERS-VOI-aligned metadata and reproducibility
   fields for experimental and future stable frontier methods.

## Out of Scope

1. Treating all frontier methods as stable immediately.
2. Requiring external stakeholder or preference elicitation packages in the base
   install.
3. Replacing the existing EVPI, EVPPI, EVSI, ENBS, structural, NMA, adaptive,
   sequential, portfolio, heterogeneity, dominance, or CEAF APIs.
4. Making `voiage` a full HTA workflow manager; the focus remains VOI analysis
   and decision-theoretic uncertainty value.
