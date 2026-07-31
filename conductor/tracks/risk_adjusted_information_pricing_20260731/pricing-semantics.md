# Pricing semantics and result records

## Canonical value functions

For a sure transfer `c`, the runtime exposes no public generic optimizer but
internally evaluates:

```text
baseline(c) = max_action E[U(initial_wealth + payoff(action, state) + c)]
informed(c) = sum_signal max_action sum_state joint(signal, state)
              * U(initial_wealth + payoff(action, state) + c)
```

The maximizing action tie set is recomputed at every `c`. Holding the
zero-price action fixed during BPI or SPI search violates the contract.

## Typed measure records

Every measure record contains `measure`, `status`, `value`, `unit`,
`direction`, `normalization`, and `diagnostics_ref`.

| Measure | Definition | Unit | Available when |
|---|---|---|---|
| EUI | `informed(0) - baseline(0)` | declared utility unit | both expected utilities are finite |
| CEI | `U^-1(informed(0)) - U^-1(baseline(0))` | payoff unit | both expected utilities are inside the inverse domain |
| BPI | root of `informed(-b) - baseline(0)` | payoff unit | a valid non-negative acquisition-price bracket converges |
| SPI | root of `baseline(s) - informed(0)` | payoff unit | a valid non-negative surrender-price bracket converges |
| PPI | `(informed(0)-baseline(0))/(informed(0)-U(z0))` | dimensionless | `z0` is a valid floor and the ratio is in `[0,1]` |

Measure status is one of `available`, `unavailable`, or `failed`. A failed or
unsupported measure carries no numeric value and includes a stable reason; a
genuine zero remains `available` with value `0.0`.

## Root result contract

Each price root record returns:

- `status`: `converged`, `zero_boundary`, `not_bracketed`, `utility_domain`,
  `non_monotone`, `max_iterations`, or `max_evaluations`;
- lower and upper price bounds, estimate and final bracket width when a root
  is available;
- raw utility residual at the estimate;
- iteration and evaluation counts;
- complete action tie sets at both bounds;
- `policy_switched` plus the ordered set of observed representative-policy
  transitions; and
- termination reason and the exact solver settings used.

Convergence is based on
`width <= absolute_price_tolerance + relative_price_tolerance*abs(estimate)`.
Utility residual is diagnostic because it changes under utility
normalization. Expansion is deterministic, bounded by `maximum_price`, and
cannot step outside the utility domain.

## Price direction and cost location

- BPI direction is `pay_to_acquire_information` and uses `informed(-b)`.
- SPI direction is `receive_to_surrender_information` and uses `baseline(s)`.
- The only supported cost location is `ex_ante_sure_transfer`.
- State-, signal-, action-, tax-, discount- or financing-dependent transfers
  must be represented in terminal payoffs before this method is called.

This contract prevents a maximum-acquisition-price helper with a different
cash-flow location from being treated as BPI without an explicit conversion.

## Comparability matrix

| Comparison | Same problem | Across problems sharing utility |
|---|---|---|
| EUI vs CEI | same ordering | same ordering only for affine utility |
| EUI vs BPI | same ordering for affine/exponential; otherwise not guaranteed | same ordering only for affine utility |
| CEI vs BPI | same ordering for affine/exponential; otherwise not guaranteed | same ordering for affine/exponential utility |
| CEI vs SPI | same ordering | requires common monetary and stakeholder basis |
| fixed-floor PPI | same ordering as EUI | requires the same valid floor and utility normalization |

The runtime returns the applicable rule IDs and missing comparability fields.
It does not infer an organizational utility from multiple stakeholder results.

## Frozen analytical examples

### Affine reduction

For positive-affine utility, the result must satisfy
`EUI = slope * information_value` and
`CEI = BPI = SPI = information_value`. The information value is EVPI only
when `information_kind` is `clairvoyant`; it is the corresponding sample/signal
value for a finite signal.

### Nonlinear buy/sell asymmetry

For log utility, initial wealth `10`, a safe action paying `0`, and a risky
action paying `5` with probability `0.8` and `-9` with probability `0.2`, under
clairvoyance, the independent reference values are:

```text
EUI = 0.3243720865 utility units
CEI = 3.8316186722 payoff units
BPI = 3.7521886610 payoff units
SPI = 3.4085030261 payoff units
```

These values intentionally disprove a general CEI/BPI/SPI equality claim.

### Cross-decision ranking reversal

For exponential utility `U(x)=-exp(-x/3000)`, the frozen two-department
reference demonstrates that EUI and BPI need not rank decision problems alike:

```text
department_1 EUI = 0.1417343; CEI/BPI = 458.525
department_2 EUI = 0.1231203; CEI/BPI = 622.700
```

The result therefore reports both within-problem availability and
cross-problem ranking conditions.

## VoC presentation

The `voc` presentation contains `presentation_label: "voc"`, a selected
measure name, and the canonical result reference. It does not add another
numeric field. A request to display monetary `evpi` from a nonlinear utility
result fails with an explicit `affine_reduction_required` diagnostic.
