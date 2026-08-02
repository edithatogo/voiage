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

At each transfer, feasibility is local to the probability support being
optimized. The baseline support is the set of states with positive state
probability; the conditional support for signal `s` is the set with positive
`joint(s, state)`. An action whose terminal outcome is outside the named
utility's domain on that support is an explicitly diagnosed infeasible action,
not an action with an artificial utility of negative infinity. Optimization
continues over the remaining feasible actions. The evaluation returns
`utility_domain` only if a positive-probability baseline or signal-conditional
scope has no feasible action. Zero-weight states and zero-probability signals
cannot create a domain failure.

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
  `non_monotone`, `discontinuous_no_root`, `max_iterations`, or
  `max_evaluations`;
- lower and upper price bounds, estimate and final bracket width when a root
  is available;
- raw utility residual at the estimate;
- iteration and evaluation counts;
- complete lexicographically sorted signal-to-policy mappings at every
  evaluation, the estimate, and both final bounds;
- deterministic action-domain exclusions at every evaluation, sorted by
  signal ID, action ID and state ID, with `signal_id` (null for baseline),
  `action_id`, failed positive-support `state_ids`, and reason
  `utility_domain`;
- `policy_switched` plus the ordered set of observed complete-policy
  transitions, each retaining the transfer and prior and next signal-policy
  mappings; and
- termination reason and the exact solver settings used.

The price-width criterion is
`width <= absolute_price_tolerance + relative_price_tolerance*abs(estimate)`;
convergence additionally requires `abs(residual) <= utility_tolerance`.
Utility residual is diagnostic because it changes under utility
normalization. Expansion is deterministic and bounded by `maximum_price`.
Crossing an individual action's domain boundary records an exclusion; crossing
the boundary of the final feasible action in a positive-probability policy
scope terminates with `utility_domain` and no fabricated price.
Bracket width alone is not evidence of equality: convergence requires the
price-width criterion and an evaluated residual within utility tolerance.
Floating-point stagnation without both returns `discontinuous_no_root` and no
price; `non_monotone` is reserved for an observed monotonicity violation.

`policy_switched` is derived from the complete signal-policy mappings, not only
the selected representatives. It is therefore true when any signal's tied
action enters or leaves its tie set even if every representative is unchanged.

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

At the BPI, the risky action is outside the log domain in the adverse signal's
positive-support state, while the safe action remains feasible there. The
adverse/risky exclusion is recorded explicitly and the conditional optimizer
selects safe; this is a valid contingent policy, not silent conversion of an
undefined utility to negative infinity. If every action for that signal were
outside the domain, BPI evaluation would fail with `utility_domain`.

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
