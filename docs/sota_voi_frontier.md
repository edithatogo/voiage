# SOTA VOI Method Frontier

## Purpose

`voiage` should aim beyond parity with existing VOI packages. The core method
set is now broad enough that the next differentiator is a frontier layer:
methods that make uncertainty analysis useful for multi-stakeholder,
distributional, implementation, and robust decision contexts.

## Already Covered

The current library already covers the standard and advanced VOI families that
most users expect:

- EVPI, EVPPI, EVSI, and ENBS
- CEAC, CEAF, dominance, and VOI curve plotting
- structural uncertainty VOI
- network-meta-analysis VOI
- adaptive, calibration, observational, sequential, and portfolio VOI
- Value of Heterogeneity

## Frontier Method Families To Add

### Value Of Perspective

Value of Perspective (VOP) treats the decision perspective as an explicit
analysis dimension rather than a hidden modelling assumption. A decision problem
can be evaluated simultaneously from payer, societal, patient, provider,
regulator, equity-weighted, or custom stakeholder perspectives.

The first contract should represent net benefit as:

```text
sample x strategy x perspective
```

The result should include:

- optimal strategy by perspective
- expected net benefit by strategy and perspective
- cross-perspective regret matrix
- value of switching perspective
- robust or consensus strategy under perspective weights
- Pareto/non-dominated strategy set across perspectives

This is different from ordinary sensitivity analysis because it compares the
decision consequences of alternative objective functions side by side.

Current fixture-backed implementation:

- Python API: `voiage.methods.value_of_perspective`
- High-level wrapper: `DecisionAnalysis.value_of_perspective`
- CLI: `voiage calculate-perspective perspective_surface.json`
- Plotting: `voiage plot-perspective-regret perspective_surface.json`
- Contract scaffold: `specs/frontier/perspective/v1/`
- Deterministic screening-program fixtures: `specs/frontier/perspective/v1/fixtures/`
- Frontier fixture registry: `specs/frontier/fixtures/manifest.json`
- Frontier fixture validator: `scripts/validate_frontier_contract.py`

## Experimental API Warnings

The frontier methods in this track span implemented, experimental, and planned
surfaces. Their schemas, fixtures, and reporting payloads are meant to anchor
the current contract, but the API surface can still change as the remaining
frontier families are implemented and cross-language parity is added.

The fixture-backed manifests should be treated as the current compatibility
baseline for `Value of Perspective`, preference, validation, threshold,
distributional/equity, and implementation-adjusted VOI. External consumers
should key against the versioned schemas and registry manifest rather than
assuming the methods are stable.

### Distributional And Equity-Weighted VOI

Distributional and equity-weighted VOI should connect VOI to equity-informed
economic evaluation. The contract should support subgroup distributions, equity
weights, inequality summaries, and social-welfare functions. This extends the
existing Value of Heterogeneity surface rather than replacing it.

Current experimental implementation:

- Python API: `voiage.methods.value_of_distributional_equity`
- High-level wrapper: `DecisionAnalysis.value_of_distributional_equity`
- Experimental result payload with subgroup-optimal summaries and maturity
  metadata
- Regression coverage for curated exports and input validation
- Deterministic fixtures: `specs/frontier/distributional/v1/fixtures/`
- Frontier fixture registry: `specs/frontier/fixtures/manifest.json`
- Frontier fixture validator: `scripts/validate_frontier_contract.py`

### Implementation-Adjusted VOI

Implementation-adjusted VOI should account for uptake, adherence, coverage,
implementation delay, and implementation uncertainty. This family connects
`voiage` to future `innovate` artifacts and to value-of-implementation analysis.

Current experimental implementation:

- Python API: `voiage.methods.value_of_implementation`
- High-level wrapper: `DecisionAnalysis.value_of_implementation`
- Experimental result payload with implementation multiplier, adjusted
  expected net benefits, and maturity metadata
- Regression coverage for curated exports and input validation
- Deterministic fixtures: `specs/frontier/implementation/v1/fixtures/`
- Frontier fixture registry: `specs/frontier/fixtures/manifest.json`
- Frontier fixture validator: `scripts/validate_frontier_contract.py`

### Preference Heterogeneity And Individualized Care

Preference heterogeneity separates uncertainty in clinical effects and costs
from uncertainty in patient or stakeholder preferences. The runtime surface is
implemented and exposed through the preference CLI entrypoint, and the
fixture-backed contract under `specs/frontier/preference/v1/` captures the
current comparison shape so users can compare value of individualized care and
value of preference information when preference elicitation could change the
decision.

The contract under `specs/frontier/preference/v1/` uses the same side-by-side
profile pattern as Value of Perspective so preference profiles can be compared
directly rather than collapsed into a single average profile.

Typical usage keeps the profile axis explicit:

```python
from voiage.analysis import DecisionAnalysis
from voiage.methods.preference import PreferenceProfile, PreferenceProfileSet

result = DecisionAnalysis(...).value_of_preference(
    PreferenceProfileSet([
        PreferenceProfile("payer"),
        PreferenceProfile("patient"),
    ])
)
```

The CLI companion is `voiage calculate-preference`, which is useful when you
want to validate the same runtime surface from a fixture-backed JSON payload.

### Model-Validation VOI

Model-validation VOI should cover the value of external validation, model
discrepancy reduction, and prediction-model validation when a prediction model
or model class is itself decision-relevant.

The contract scaffold for this surface now lives under
`specs/frontier/validation/v1/`. It uses the same side-by-side profile pattern
as Value of Perspective so that future implementations can compare validation
scenarios and discrepancy-reduction strategies directly. The current state is
implemented runtime plus fixture-backed conformance, and the CLI entrypoint is
`calculate-validation`.

### Threshold, Tipping-Point, And Robust VOI

Threshold and tipping-point analyses quantify how close a decision is to
reversal under willingness-to-pay thresholds, budget thresholds, evidence
thresholds, or policy constraints. Robust VOI should cover ambiguity sets,
multi-objective trade-offs, and decision value under model ambiguity.

The contract scaffold for this surface now lives under
`specs/frontier/threshold/v1/`. It follows the same profile-based comparison
pattern as Value of Perspective so that future implementations can compare
threshold, tipping-point, and robustness scenarios side by side. The current
state is implemented runtime plus fixture-backed conformance, and the CLI
entrypoint is `calculate-threshold`.

### Dynamic Real-Options VOI

Dynamic real-options VOI should extend sequential VOI when delay,
irreversibility, implementation timing, or policy lock-in materially changes the
value of information.

### Adjacent Frontier Extensions

The current implementation track should prioritize the methods above, but the
SOTA roadmap should also keep these adjacent families visible:

- causal-identification, transportability, and external-validity VOI, where the
  question is whether better causal evidence would change the decision in a
  target population
- data-quality, missingness, measurement-error, data-acquisition, privacy, and
  linkage VOI, where the information source itself has operational constraints
- computational VOI and value of model refinement, where the decision is
  whether more simulation, calibration, model comparison, or metamodel training
  is worth its cost
- expert-elicitation and evidence-synthesis design VOI, where the decision is
  whether to elicit uncertain parameters, resolve conflicting sources, or invest
  in a better synthesis process

## Track Ownership

The implementation plan lives in:

`conductor/archive/sota-voi-frontier_20260429/`

Value of Perspective should be implemented first because it is clearly
differentiating, directly connected to the user's work, and can reuse the
existing net-benefit-first architecture.

## References And Starting Points

- ISPOR Value of Information Analysis Emerging Good Practices Task Force
  Reports 1 and 2 define the current good-practice baseline for VOI research
  decisions and analytical methods:
  <https://www.sciencedirect.com/science/article/pii/S109830152030036X>
  <https://www.sciencedirect.com/science/article/pii/S1098301520300279>
- CHEERS-VOI provides the reporting layer for VOI analyses performed alongside
  economic evaluations and requires VOI-specific reporting items in addition to
  the CHEERS 2022 baseline:
  <https://www.ispor.org/publications/journals/value-in-health/abstract/Volume-26--Issue-10/Consolidated-Health-Economic-Evaluation-Reporting-Standards---Value-of-Information-%28CHEERS-VOI%29--Explanation-and-Elaboration>
- The Value of Heterogeneity framework motivates static and dynamic value from
  subgroup-specific decision making and helps position Value of Perspective
  relative to existing subgroup methods:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4232328/>
- Value of information on preference heterogeneity and individualized care
  motivates treating preferences as decision-relevant uncertainty rather than
  only clinical or cost uncertainty:
  <https://pubmed.ncbi.nlm.nih.gov/17409362/>
- Distributional cost-effectiveness analysis motivates explicit equity and
  distributional impact handling:
  <https://academic.oup.com/book/29892>
- Implementation-adjusted EVSI work motivates extending standard EVSI when
  implementation is delayed, incomplete, or evidence-dependent:
  <https://journals.sagepub.com/doi/10.1177/0272989X211073098>
- The `vop_poc_nz` preprint demonstrates a Python proof-of-concept for
  distributional cost-effectiveness and Value of Perspective analysis:
  <https://arxiv.org/abs/2512.03596>

## Reporting Baseline

CHEERS-VOI is the reporting floor for this track. Every frontier method should
be able to surface the information needed to reproduce and assess the analysis:

- decision context and comparators
- perspective and population metadata
- model structure and parameterization
- uncertainty characterization and probabilistic analysis settings
- VOI method family, estimator choice, and maturity label
- reproducible outputs and diagnostics

The current experimental frontier APIs now emit structured reporting payloads
with CHEERS-VOI metadata, method maturity, diagnostics, and reproducibility
placeholders. Future frontier work should keep expanding that reporting surface
until the checklist is satisfied end to end, including the remaining frontier
families beyond Value of Perspective, distributional/equity VOI, and
implementation-adjusted VOI.
