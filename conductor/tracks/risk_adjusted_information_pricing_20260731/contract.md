# Frozen expected-utility information-pricing contract

## 1. Decision problem

The v1 input is a finite scalar terminal-payoff decision problem:

- `decision_problem_id` and `stakeholder_scope_id` are non-empty stable IDs;
- `action_ids` and `state_ids` are unique non-empty strings;
- `payoffs[state, action]` are finite terminal transfers in one declared unit;
- `state_probabilities` are finite, non-negative and sum to one;
- `initial_wealth` and `reference_wealth` are finite in the payoff unit;
- currency-valued problems declare `currency` and `price_date`;
- `information_kind` is `clairvoyant` or `finite_signal`;
- a finite-signal model supplies unique `signal_ids` and a non-negative joint
  `signal_state_probabilities[signal, state]` matrix that sums to one and whose
  state marginals equal `state_probabilities` within declared tolerance;
- clairvoyance is represented by one signal per state and the diagonal joint
  matrix, not a distinct kernel;
- ties return the complete lexicographically sorted action-ID set and select
  its first member only as the representative policy.

Zero-probability states and signals are retained for schema identity but do
not affect expected utility. A zero-probability signal has an empty policy and
must never trigger posterior division.

## 2. Named utility descriptors

Only tagged, serializable, strictly increasing utilities are accepted:

| Family | Parameters | Canonical utility | Domain |
|---|---|---|---|
| `affine` | `slope > 0`, finite `intercept` | `slope * x + intercept` | all finite `x` |
| `exponential` | `risk_tolerance > 0`, `reference_wealth` | `-exp(-(x-reference_wealth)/risk_tolerance)` | all finite `x` with finite evaluation |
| `log` | `reference_wealth > 0` | `log(x/reference_wealth)` | `x > 0` |
| `power` | `risk_aversion > 0`, `risk_aversion != 1`, `reference_wealth > 0` | `((x/reference_wealth)^(1-risk_aversion)-1)/(1-risk_aversion)` | `x > 0` |

The result repeats the family, parameters, normalization and domain. Arbitrary
callbacks are unsupported because monotonicity, inversion, portability and
deterministic serialization cannot be assured.

## 3. Canonical policy values

Let `Y[a,w] = initial_wealth + payoffs[w,a]`, utility `U`, state probability
`p[w]`, and joint signal/state probability `pi[s,w]`. For a sure transfer `c`
applied before utility, define:

```text
B(c) = max_a sum_w p[w] * U(Y[a,w] + c)
I(c) = sum_s max_a sum_w pi[s,w] * U(Y[a,w] + c)
```

Every evaluation re-optimizes the current or signal-conditional policy. The
joint weights are used directly so zero-probability signals never require
posterior division. `I(c) >= B(c)` within numerical tolerance because the
uninformed policy remains feasible after information.

The only v1 information-cost location is `ex_ante_sure_transfer`: the price is
the same additive transfer in every terminal state before utility evaluation.
State-, signal- or action-dependent costs must be included in the payoff model
and cannot use the v1 price equations silently.

## 4. Measures and direction

The canonical acquisition direction is `uninformed_to_informed`:

```text
EUI = I(0) - B(0)
CEI = inverse_U(I(0)) - inverse_U(B(0))
BPI = b >= 0 such that I(-b) = B(0)
SPI = s >= 0 such that B(s) = I(0)
PPI = (I(0) - B(0)) / (I(0) - U(z0))
```

`z0` is a declared terminal-outcome floor in the payoff unit and must be no
greater than every positive-probability terminal outcome. PPI is available
only when its denominator is positive and its value lies in `[0, 1]` within
tolerance. Its equivalent mixture relation is
`PPI*U(z0) + (1-PPI)*I(0) = B(0)`.

EUI has the declared canonical utility unit. CEI, BPI and SPI have the payoff
unit. PPI is dimensionless. Signed computed values are retained. A tolerance-
zero root returns a converged zero result; no other failure or negative value
is clamped to zero.

## 5. Deterministic root solving

BPI and SPI use safeguarded bisection, not derivatives across policy switches.
The input declares positive `initial_upper`, `expansion_factor > 1`,
`maximum_price >= initial_upper`, positive absolute and relative price
tolerances, and positive `maximum_iterations` and `maximum_evaluations`.

The solver:

1. evaluates the zero-price objective;
2. returns a converged zero boundary only when that objective is within the
   declared utility tolerance;
3. expands the upper bound geometrically without exceeding `maximum_price`;
4. prevalidates utility domains at every evaluated transfer;
5. requires a sign-changing bracket;
6. bisects until the bracket-width tolerance is met; and
7. returns lower/upper bounds, estimate, width, raw utility residual,
   iterations, evaluations, action tie sets at every evaluated transfer and
   both final bounds, ordered complete tie-set transitions, policy-switch
   evidence, convergence and one of `converged`, `zero_boundary`,
   `not_bracketed`, `utility_domain`, `non_monotone`, `max_iterations`, or
   `max_evaluations`.

Unavailable prices use a discriminated status record. The maximum search
price, a null, or a fabricated zero must not be returned as an estimate.
`policy_switched` is true when any complete action tie set changes between
ordered evaluations, even if the lexicographic representative remains the
same. Every transition records the transfer, prior and next sorted tie sets,
and their representative action IDs.

## 6. Comparability

The result separately records numeric comparability and ranking equivalence.

- EUI changes under positive-affine rescaling of utility; CEI, BPI, SPI and
  fixed-floor PPI do not.
- Within one problem, EUI, CEI, SPI and fixed-floor PPI are ordinally
  equivalent. BPI joins them for affine or exponential utility.
- Across problems sharing one declared utility, BPI and EUI rankings agree
  only for affine utility; BPI and CEI rankings agree for affine or
  exponential utility; EUI and CEI rankings agree only for affine utility.
- Cross-problem monetary-price comparison requires common payoff/currency
  unit, price date, wealth basis, utility identity, normalization,
  stakeholder/organizational scope and information convention.
- Different stakeholder utilities remain separate unless an explicit
  organizational utility over joint terminal outcomes is supplied. Weighted
  stakeholder EUI is not a monetary buying price.

## 7. VoC presentation and affine reduction

`voc` is a presentation label on this result with
`information_kind = clairvoyant`. It selects EUI, CEI, BPI, SPI or PPI for
display but does not serialize a second scalar and does not call another
kernel.

For `U(x) = slope*x + intercept`, `slope > 0`:

```text
EUI = slope * monetary_information_value
CEI = BPI = SPI = monetary_information_value
```

The monetary information value is EVPI only for clairvoyance and EVSI for an
imperfect finite signal. Nonlinear utility never receives an unconditional
monetary-EVPI alias. Under exponential utility, CEI/BPI/SPI equality is a
translation-invariance property; EUI remains on the declared utility scale.

## 8. Result envelope and maturity

The deterministic v1 result contains method/schema versions; input digest and
provenance; utility and decision descriptors; current and informed expected
utilities/certainty equivalents; current and per-signal policy tie sets;
EUI/CEI/BPI/SPI/PPI typed measure records; root diagnostics; affine-reduction
status; comparability state/reasons; backend identity; and reporting metadata.
Unknown fields, NaN and infinity are rejected. IDs and diagnostic collections
serialize in canonical order.

Repository delivery is `experimental` until the separate scientific-review
gate approves stable promotion and installed binding evidence supports any
stronger parity claim.
