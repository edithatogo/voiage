# Deterministic sensitivity analysis reference and boundary review

Issue: [#556](https://github.com/edithatogo/voiage/issues/556), with governed
delivery subissues [#724](https://github.com/edithatogo/voiage/issues/724)–
[#728](https://github.com/edithatogo/voiage/issues/728).

## Frozen initial contract

The experimental v1 contract freezes a baseline parameter vector and decision
metric before evaluation. One-way analysis varies exactly one declared
coordinate over an ordered grid while holding every other coordinate at the
baseline. Two-way analysis evaluates declared pairs while holding remaining
coordinates fixed, subject to an explicit feasible mask or path when a full
Cartesian surface would violate logical or correlation constraints. A scenario
is a named bundle of deterministic overrides or structural assumptions, not a
probability-weighted sample.

Inputs declare parameter units, output unit, maximize/minimize direction,
compared alternatives, range provenance, coordinate order and baseline
provenance. Results retain raw alternative outputs as well as
direction-normalized increments. Every point returns the complete co-optimal
set under declared absolute and relative tolerances; a canonical name-order
presentation policy cannot remove ties.

The initial threshold result returns every exact evaluated tie, tie plateau and
adjacent bracket where the complete optimal set or declared contrast sign
changes. Multiple, absent and discontinuous switches are valid. It does not
interpolate or extrapolate a unique root. A later optional root estimator must
label its result estimated and record continuity/monotonicity assumptions,
method, tolerance, iterations and bracket.

Tornado ranking uses a named formula. The initial default is the evaluated-grid
range `max(metric) - min(metric)`, with low-coordinate, high-coordinate and
interior extrema retained separately. It is not a probability, global
importance measure or value-of-information result.

## Independent authoritative references

- The ISPOR-SMDM Task Force report on parameter estimation and uncertainty
  distinguishes deterministic variation from joint probabilistic sampling,
  describes one-way tornado ranges, scenarios, two-way decision regions and
  threshold values:
  <https://www.ispor.org/docs/default-source/resources/outcomes-research-guidelines-index/model_parameter_estimation_and_uncertainty-6.pdf?sfvrsn=8bc10c8e_0>.
- Canada's NACI economic-evaluation guideline defines DSA as varying one or
  more prespecified inputs while holding others fixed, separates structural
  scenarios, and requires logical/correlation relationships to be preserved:
  <https://www.canada.ca/en/public-health/programs/guidelines-economic-evaluation-vaccine-programs-canada-stakeholder-consultation/guidelines-document.html>.
- Australia's PBAC uncertainty guidance requires base-case, alternatives or
  ranges and incremental outcomes to be tabulated, uses tornado diagrams for
  relative base-case effects, and treats PSA as an additional distinct
  analysis:
  <https://pbac.pbs.gov.au/section-3a/3a-9-uncertainty-analysis-model-inputs-and-assumptions.html>.
- The European Commission Joint Research Centre distinguishes local
  one-at-a-time analysis at nominal values from global sensitivity analysis
  over multidimensional variation and interactions:
  <https://joint-research-centre.ec.europa.eu/sensitivity-analysis-samo/methods_en>.
- The ISPOR value-of-information report and Strong et al. EVPPI method define
  EVPPI through probabilistic decision analysis under a joint parameter law,
  not through deterministic tornado widths or switch points:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/> and
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4819801/>.

Search limit: these sources govern the initial terminology, boundary and test
oracle; this record is not a systematic review.

## Fail-closed and test boundary

The runtime must reject missing or non-finite baselines/outputs, empty grids,
duplicate or unknown coordinates, unit conflicts, malformed callback shapes,
callback exceptions and unsupported partial/extrapolated results. Tests cover
linear references, nonlinear interior extrema, multiple/exact/no switches,
discontinuities, flat tie plateaus, maximize/minimize reversal, permutation
invariance, tolerance boundaries, feasible masks, repeatability and attempted
NaN/Inf inputs. The DSA contract accepts no distributions, probability weights,
random seeds or EVPPI labels and emits no probabilistic, global, causal or VoI
claim.
