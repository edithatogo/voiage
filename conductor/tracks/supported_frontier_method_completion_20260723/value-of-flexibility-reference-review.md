# Value of Flexibility reference and boundary review

Issue: [#559](https://github.com/edithatogo/voiage/issues/559).

## Frozen initial estimator

The first executable slice treats decision stages as mutually exclusive timing
scenarios with declared probabilities `w[t]`. Every payoff is expressed in one
declared value unit after a common discount/cost adjustment. For adjusted value
`U[s,t]`, the unconstrained reference is:

\[
V_{flex}=\sum_t w_t\max_{s\in F_t}U_{s,t},\qquad
V_{commit}=\max_{s\in C}\sum_t w_tU_{s,t},\qquad
VoF=V_{flex}-V_{commit}.
\]

The commitment set `C` must be non-empty and feasible in every scenario, so it
is a subset of every flexible set `F_t`. This establishes nonnegativity without
clipping and makes the result invariant to strategy ordering. Sequential
lifecycle-period aggregation and transition-constrained policy paths are not
silently coerced into this scenario contract; a later dynamic-programming
extension must declare them separately.

Numerical ties use a frozen `canonical-lexicographic` policy: the runtime
returns every tied strategy in canonical name order and presents the first name
as the selected path or commitment baseline. This prevents input ordering from
changing a versioned result while retaining the complete tie set for audit.

## Independent references

- Grutters et al., [Real Options Analysis in Proton Therapy](https://pmc.ncbi.nlm.nih.gov/articles/PMC3248774/), provides enumerable adopt, delay and trial strategies with explicit reversibility and research cost.
- Marescot et al., [A primer on stochastic dynamic programming](https://doi.org/10.1111/2041-210X.12082), provides the independent finite-horizon policy-enumeration/DP assurance pattern required for a later transition-aware extension.
- Cook et al., [Lessons on the value of research infrastructure](https://pmc.ncbi.nlm.nih.gov/articles/PMC8013164/), distinguishes real-option value from learning/quasi-option value and supports an explicit decomposition boundary.

Search limit: these issue- and implementation-specific sources establish the
initial contract and tests; they are not represented as a systematic review.

## Fail-closed boundary

The initial runtime rejects missing/extra named weights, non-increasing timing,
incomparable or empty units, an infeasible commitment set, unsupported stage
semantics and a declaration that information value is already embedded. It
returns an information-value component of zero and does not label VoF as EVPI,
EVSI, value of control, robustness or model-uncertainty value.

The v1 input must declare scenario probabilities and deterministic provenance.
Non-zero discount, irreversibility and lock-in controls fail closed: the first
slice does not assign units or policy-dependent meanings to those controls, so
accepting them would imply unsupported semantics (and a scenario-common
lock-in shift would cancel from VoF). The result carries named stage/strategy
axes. Since timing scenarios are mutually exclusive rather than a sequential
path, `exercise_decisions` is explicitly unsupported and adjacent ordered
scenario choice changes are only a diagnostic.

The pre-existing dynamic-real-options compatibility envelope retains its
historical zero-time default and first-in-input tie presentation. Those legacy
presentation rules are not inherited by the experimental versioned VoF result.
