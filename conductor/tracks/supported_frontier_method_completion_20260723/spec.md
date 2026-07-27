# Track Specification: Supported Frontier Method Completion

## Overview

Audit, complete, consolidate, or correctly classify every broader VOI family.

## Requirements

Cover structural/model form, NMA, calibration, observational, sequential,
adaptive, real options, portfolio/capacity, heterogeneity, individualized care,
preference, equity/distributional, implementation, validation, replication,
obsolescence, threshold, ambiguity/shift, causal/transportability, correlation,
data quality, computational, monitoring, expert synthesis, MCDA, strategic,
privacy/federated, regulatory, and interoperability families.

Each family receives a formal estimand, typed contract, deterministic runtime,
diagnostics, failure policy, fixture, properties, reference evidence, maturity,
and cross-language disposition. Duplicates use versioned aliases and
deprecations. Placeholders, global RNG, broad exception swallowing, and
optional-dependency leakage block promotion.

### Governed method-gap workstreams

The following gaps are native subissues of GitHub #318 and remain owned by this
track. They are not separate Conductor tracks:

| Issue | Method ID | Method family | Required disposition |
| --- | --- | --- | --- |
| [#556](https://github.com/edithatogo/voiage/issues/556) | `deterministic-sensitivity-analysis` | Deterministic Sensitivity Analysis (DSA) | Implement an explicit one-way, multi-way, and scenario-analysis contract, or record a scientifically reviewed exclusion. Keep it distinct from probabilistic sensitivity analysis (PSA), variance-based global sensitivity analysis, and VOI estimands. |
| [#557](https://github.com/edithatogo/voiage/issues/557) | `value-of-distributional-information` | Value of Distributional Information (VDI) | Implement information value for uncertainty over distributional families or assumptions, or record a reviewed exclusion. Keep it distinct from distributional/equity VOI over population outcomes. |
| [#558](https://github.com/edithatogo/voiage/issues/558) | `qualitative-voi` | Qualitative VoI | Provide an executable, auditable qualitative assessment contract that never invents a quantitative information value. |
| [#559](https://github.com/edithatogo/voiage/issues/559) | `value-of-flexibility` | Value of Flexibility (VoF) | Formalize the adjacent option-value estimand and its relationship to dynamic real-options and sequential decision contracts without conflating flexibility with information. |
| [#560](https://github.com/edithatogo/voiage/issues/560) | `mcda-voi` | Multi-Criteria Decision Analysis VOI (MCDA-VOI) | Implement a real decision and information-value contract for explicitly supported MCDA model families; mock-only and arbitrary weighted-score surfaces are insufficient. |

### DSA contract boundary

DSA inputs must identify the baseline parameterization, parameter names and
units, valid ranges or named scenarios, evaluation ordering, strategies,
outcomes or net benefit, and deterministic tie policy. Results must retain the
baseline decision, evaluated parameter/scenario values, outcome and incremental
outcome curves, decision-switch points, rankings, and diagnostics. Invalid
ranges, mismatched units, duplicate names, non-finite evaluations, and
unreproducible evaluator behavior fail closed. DSA must not be presented as
EVPI, EVPPI, PSA, or Sobol-style sensitivity.

### VDI contract boundary

VDI inputs must define candidate distributional models, normalized model
probabilities, parameters or draws, alternatives, utility or loss, conditioning
order, and optional information cost. Results must report the baseline and
resolved decisions, per-model conditional values, gross and net VDI, estimator
uncertainty, and provenance. Unsupported or unidentified models, invalid
probabilities, non-finite or misaligned draws, and inadequate estimator evidence
fail closed. Distributional equity and subgroup heterogeneity are not VDI.

### Qualitative VoI contract boundary

The portable assessment must version the decision, uncertainties, potential
impact, feasibility, timeliness, equity and ethics, proposed information
actions, costs or burdens, confidence, rationale, provenance, dissent, missing
information, and review state. It may return ordered information questions and
explicit recommendation classes, but it must not coerce ordinal judgements into
probabilities, utilities, currency, or a quantitative VOI. Deterministic
serialization, redaction boundaries, disagreement preservation, audit history,
accessibility, and human review are required. AI-assisted content remains
unverified until a named human review state is recorded.

### VoF contract boundary

VoF inputs must define flexible and constrained policy sets, the commitment
baseline, transitions, exercise rules, chronology, discounting, irreversibility,
lock-in, information availability, costs, and units. Results must expose the
flexible and constrained values, VoF, waiting or option components, policy
paths, exercise decisions, regret, and diagnostics. Combined analyses must
decompose option value from EVPI, EVSI, value of control, and adaptive
information value. Incomparable policies, invalid chronology, unreachable
actions, inconsistent discounting, and double counting fail closed.

### MCDA-VOI contract boundary

MCDA-VOI inputs must identify an explicitly supported decision-rule family,
criteria, units and directions, value functions, preferences or weights,
correlations, aggregation, thresholds or vetoes, alternatives, uncertainty,
information actions, normalization, and tie policy. Results must retain
baseline rankings and choices, conditional choices, criterion and preference
information values, regret or loss, rank acceptability, dominance diagnostics,
and estimator assurance. Unsupported compensatory or outranking assumptions,
incoherent scales, double counting, unidentified preferences, and missing
uncertainty fail closed.

### Cross-cutting implementation and parity

Numerical kernels belong in the Rust core when the estimand is accepted for
implementation. Python remains the primary orchestration facade; Rust, Python,
R, Julia, and Mojo capability manifests must each declare `implemented`,
`adapter`, `contract-only`, `unsupported`, or `upstream-blocked` for every
workstream. Qualitative VoI may share a Rust-owned portable schema without a
numerical kernel. No binding may advertise a method that its installed runtime
cannot execute or validate.

Each accepted method requires analytical or enumerable examples, an independent
reference, invariants and metamorphic properties, invalid/pathological
fixtures, estimator or audit assurance, deterministic serialization, registry
and capability synchronization, executable documentation, and an explicit
maturity decision. Public or external data are promotion evidence only when
rights, provenance, hashes, and coverage are recorded.

## Acceptance criteria

Every frontier module is implemented, consolidated, scaffold-only, or excluded
with evidence; #556--#560 have a reviewed disposition and their acceptance
evidence is linked from this track; no unsupported stable claim remains; public
data is required before supported/stable promotion.

## Out of scope

Promotion from fixture presence alone.
