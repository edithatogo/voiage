# MCDA information value reference and adjacent-surface review

Issue: [#560](https://github.com/edithatogo/voiage/issues/560).

## Frozen v1 decision model

The first executable contract is deliberately narrower than MCDA in general.
It supports a finite set of alternatives `A`, a finite set of criteria `K` and
a finite joint uncertain state `omega` with declared probability `p[omega]`.
For alternative `a` and criterion `k`, `x[a,k,omega]` is the predicted raw
consequence in the criterion's declared unit and direction. A value function
`v[k]`, fixed before evaluating any information action, maps that consequence
to a common dimensionless value scale:

\[
z_{a k}(\omega)=v_k\!\left(x_{a k}(\omega)\right).
\]

The value-function family, parameters, raw-unit anchors, direction and
extrapolation policy are part of the frozen decision problem. Min-max scaling
over the submitted alternatives, the Monte Carlo sample, a conditional group
or a post-information result is prohibited: it would allow the information
action to change the measuring scale as well as knowledge about the state.

The v1 aggregation family is the compensatory additive value model

\[
U(a,\omega)=\sum_{k\in K}w_k(\omega)z_{a k}(\omega),\qquad
w_k(\omega)\geq 0,\qquad \sum_{k\in K}w_k(\omega)=1.
\]

Weights are dimensionless trade-off parameters on the declared value scales,
not criterion importance labels or probabilities of criteria. Additivity is a
substantive preference-independence assumption, not a generic property of
multiple-criteria problems. Criteria must be operationally distinct; splitting
or duplicating one consequence into several criteria without re-eliciting the
preference model double counts it. The runtime must therefore identify the
approved aggregation family and reject a model whose stated interactions,
thresholds or vetoes contradict compensatory additivity.

Prediction variables and preference weights may be statistically dependent.
`p[omega]` is one joint law over both, including correlations between criteria,
alternatives and weights. Marginal arrays cannot be combined as independent
unless independence is an explicit, evidenced model assumption. The initial
exact evaluator enumerates this finite joint law; sampling and metamodel
estimators are follow-on work.

## Perfect-information estimands

Let `R` be a declared finite information partition of the joint state. Its
labels must identify exactly which latent variables become known. The current
and `R`-resolved values are

\[
V_0=\max_{a\in A}E[U(a,\omega)],
\]

\[
V_R=\sum_r P(R=r)\max_{a\in A}E[U(a,\omega)\mid R=r].
\]

The gross and net information values are

\[
VOI_R=V_R-V_0,\qquad NVOI_R=VOI_R-c_R,
\]

where `c[R]` is finite, non-negative and expressed on the same aggregate-value
scale, population basis, horizon and discount basis. A monetary research cost
cannot be subtracted from dimensionless aggregate value without a declared
conversion or willingness-to-pay contract. Gross VOI is non-negative under the
same feasible alternatives, joint law, utility construction and exact
conditional expectations; the implementation must not enforce this by
post-hoc clipping. Net VOI may be negative.

A **criterion-information action** resolves named predictive latent variables
that generate one or more `x[a,k,omega]` values while integrating the remaining
prediction and preference uncertainty. A **preference-information action**
resolves named weight latent variables while integrating remaining prediction
and preference uncertainty. Neither label licenses resolving all criterion
outcomes or all weights silently. If prediction and preference variables are
dependent, each conditional expectation uses their joint conditional law.

Let `C` and `W` denote declared criterion and preference partitions and
`C join W` their joint refinement. Their interaction is

\[
I_{C,W}=VOI_{C\vee W}-VOI_C-VOI_W
       =V_{C\vee W}-V_C-V_W+V_0.
\]

It may be positive or negative. Component information values are not assumed
additive, and summing `VOI_C` and `VOI_W` cannot substitute for evaluating the
joint partition. The result must also return the conditional increments
`V[C join W]-V[C]` and `V[C join W]-V[W]` so a consumer cannot count the same
decision switch twice. If partition `R2` refines `R1`, exact values satisfy
`V[R2] >= V[R1]`. Every partial value is bounded above by perfect information
about the complete joint state under the matched decision problem.

This v1 contract is an additive-MCDA presentation of partial perfect
information. It is not sample information: an imperfect study would require a
sampling likelihood, posterior updating and pre-posterior averaging.

## Choice, ties, regret and rank acceptability

The baseline and every conditional state return expected aggregate value for
all alternatives, the complete co-optimal set and the full ranking under one
declared absolute/relative tie rule. Canonical name order is presentation only.
It cannot turn a tie into a unique choice.

Statewise opportunity loss is

\[
L(a,\omega)=\max_{b\in A}U(b,\omega)-U(a,\omega).
\]

The result retains statewise and probability-weighted regret for the baseline
policy and each information-conditioned policy. Regret is computed from the
same aggregate-value matrix as VOI; a separately normalized or clipped regret
surface would not be comparable.

For diagnostic rank acceptability, let `G(a,omega)` be the complete tie group
containing `a`, occupying ordered rank positions `q` through `q+m-1`. The v1
fractional convention assigns `1/m` to each occupied position for that
alternative in that state. Thus

\[
b_a^r=\sum_\omega p_\omega\,
       \frac{\mathbf 1\{r\text{ is occupied by }G(a,\omega)\}}
            {|G(a,\omega)|}.
\]

The complete tie groups are returned alongside this matrix so fractional
presentation cannot hide indifference. Rank acceptability describes the
distribution of ranks under the submitted joint law. It is not an information
value, a posterior probability that an alternative is truly best, or evidence
that preferences are identified.

## Expected and statewise Pareto diagnostics

Pareto diagnostics use the direction-normalized criterion value vector `z`,
not heterogeneous raw criterion units and not the weighted aggregate alone.
Alternative `a` statewise-dominates `b` in state `omega` when

\[
z_{ak}(\omega)\geq z_{bk}(\omega)\quad\text{for every }k,
\]

with a strict inequality for at least one criterion. The statewise result
reports dominance and non-dominated sets separately for every state.

The expected-value vector is

\[
\bar z_{ak}=E[z_{ak}(\omega)].
\]

Expected dominance applies the same componentwise rule to `bar(z)`. Expected
and statewise dominance are different diagnostics and neither may be inferred
from the other. The output names the expectation law and tie tolerance used.
A cost-effectiveness frontier, a weighted-score optimum and a union of
statewise non-dominated alternatives are not substitutes for these definitions.

## Units, invariants and provenance

Each criterion records raw units, direction, value-function identifier,
parameterization, anchors, valid domain and source. Aggregate values, regret,
gross VOI and net VOI use the declared common value scale. Information cost
records its original unit and any conversion into that scale. Alternative,
criterion and state identifiers are unique and row alignment is explicit.

The exact contract should satisfy:

- alternative, criterion and joint-state permutation invariance;
- invariance to a raw-unit conversion when the corresponding value function
  is transformed so every `z[a,k,omega]` is unchanged;
- invariance to any equivalent weight/value reparameterization that preserves
  every aggregate `U[a,omega]` exactly;
- zero gross value when the same complete co-optimal set applies in every
  information state;
- monotonicity under partition refinement and the matched full-information
  upper bound;
- exact decomposition of joint value and interaction without assumed
  additivity; and
- probability, rank-acceptability and conditional-value reconciliation within
  declared arithmetic tolerances.

Positive scaling of the complete aggregate-value matrix and information cost
scales all value and regret outputs by the same factor; adding one
action-independent constant in every state cancels from VOI and regret. These
properties do not authorize rescaling individual criteria without preserving
the elicited trade-offs.

Provenance includes decision and model revision, alternative and criterion
definitions, data and transformation sources, value-function and weight
elicitation sources, joint-probability source, dependence assumptions,
normalization anchors, information partitions and costs, tie policy, evaluator
and software version, and any conversion to the common aggregate scale.

## Explicit v1 exclusions

The first contract rejects rather than approximates:

- outranking families such as ELECTRE or PROMETHEE, pairwise outranking
  indices, incomparability, vetoes and non-compensatory thresholds;
- AHP pairwise-comparison elicitation, eigenvector weights and consistency
  diagnostics; supplied simplex weights are inputs, not an AHP implementation;
- multiplicative, multilinear, Choquet, fuzzy, interval, credal, robust,
  maximin, minimax-regret, chance-constrained or risk-sensitive aggregation;
- endogenous, alternative-relative, draw-relative or post-information
  normalization, and unidentified value-function or weight scales;
- treating ordinal qualitative assessments, evidence grades or criterion
  importance labels as cardinal values or weights;
- imperfect/sample information, adaptive research, preference-elicitation
  process models and posterior preference learning;
- social-choice aggregation across stakeholders; one analysis has one declared
  coherent preference model, even when its parameters are uncertain;
- resource-constrained portfolio selection, endogenous feasible sets and
  multi-objective optimization over undeclared alternatives; and
- scientific or stable claims for arbitrary weighted scoring, rank robustness,
  a Pareto plot, a schema, a mock or documentation alone.

## Adjacent repository surfaces

- `tests/test_hta_integration_comprehensive.py` imports
  `MultiCriteriaDecisionAnalysis` inside a broad `try` and replaces it with
  `Mock` after `ImportError`; its configured scores and ranks are mock behavior,
  not installed MCDA or VOI evidence.
- `voiage.methods.preference.value_of_preference_information` is currently an
  alias of `value_of_preference`. It compares already aggregated net-benefit
  profiles and does not resolve uncertain MCDA weights under a joint
  criterion/preference law. Reusing its name or scalar result would conflate
  profile switching value with the preference-information estimand above.
- `voiage.methods.perspective.value_of_perspective` compares already aggregated
  perspective-specific net benefits. Its regret matrix and profile Pareto set
  are useful adjacent presentation patterns, not an MCDA value-function,
  criterion or information-partition contract.
- `voiage.methods.dominance` implements the cost-effectiveness cost/effect
  frontier and ICER rules. Those two-dimensional, domain-specific rules cannot
  serve as general expected or statewise MCDA Pareto diagnostics.
- `voiage.methods.portfolio.portfolio_voi` selects studies under budget and
  dependency constraints. It values feasible study bundles, not a finite set
  of alternatives under multiple criterion value functions.
- `voiage.methods.qualitative_information` preserves non-cardinal assessments,
  dissent and human verification. Its ordinal priorities must never be
  converted into MCDA scores or weights without a separately governed
  elicitation contract.

Shared validation, tie and serialization utilities may be reused only where
their semantics match. No adjacent public result should be wrapped or
relabelled as #560, and the additive aggregate-value calculation should have
one implementation path for criterion, preference and joint information to
avoid conditioning drift and double counting.

## Independent references

- Keeney,
  [Utility Independence and Preferences for Multiattributed Consequences](https://doi.org/10.1287/opre.19.4.875),
  derives multiattribute utility forms from independence conditions and shows
  that the additive form is a special case rather than a universal MCDA rule.
- Marsh et al.,
  [Multiple Criteria Decision Analysis for Health Care Decision Making:
  Emerging Good Practices, Report 2](https://doi.org/10.1016/j.jval.2015.12.016),
  treats problem structuring, criteria, performance measurement, scoring,
  weighting, aggregation, uncertainty and reporting as distinct MCDA steps.
- Haag and Chennu,
  [Assessing whether decisions are more sensitive to preference or prediction
  uncertainty with a value of information approach](https://doi.org/10.1016/j.omega.2023.102936),
  applies partial perfect information to predictive and preference parameters
  under a joint expected-utility construction and notes that dependence must be
  retained in conditional analysis.
- Heath et al.,
  [Value of Information Analytical Methods: ISPOR Task Force Report
  2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/),
  gives the conditional-expectation and conditional-optimization construction
  for perfect and partial perfect information and the matched-current-decision
  comparator.
- Samson, Wirth and Rickard,
  [The value of information from multiple sources of uncertainty in decision
  analysis](https://doi.org/10.1016/0377-2217(89)90163-X),
  establishes that values for multiple information sources are generally
  non-additive, motivating explicit joint value and interaction results.
- Tervonen and Lahdelma,
  [Implementing stochastic multicriteria acceptability
  analysis](https://doi.org/10.1016/j.ejor.2005.12.037),
  defines rank-acceptability analysis for uncertain criterion measurements and
  preferences. The v1 result uses rank acceptability only as a diagnostic and
  freezes its tie convention independently of the VOI estimand.
- Azondékon and Martel,
  [Value of additional information in multicriterion analysis under
  uncertainty](https://doi.org/10.1016/S0377-2217(98)00102-7),
  identifies the difficulty of transporting scalar expected-information value
  into outranking methods with heterogeneous evaluations and differently
  scaled study costs. That is a primary reason v1 excludes outranking and
  requires a common aggregate scale before cost subtraction.

Search limit: these sources establish the additive-model assumptions,
prediction/preference conditioning, information interaction, rank diagnostic
and method-family boundaries for an initial exact contract. This record is not
a systematic review. Independent scientific and practitioner approval,
empirical elicitation validity, imperfect-information methods, polyglot
execution and stable promotion remain separate gates.
