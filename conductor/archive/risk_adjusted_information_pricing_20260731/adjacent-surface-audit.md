# Adjacent surface and VoC disposition audit

## Audit scope

The audit searched the Python runtime, Rust numerical crates, stable and
frontier specifications, fixtures, CLI, DecisionAnalysis, R/Julia bindings and
tests for expected-utility information pricing, buying/selling prices,
certainty equivalents, probability prices, maximum acquisition prices,
clairvoyance, EVPI, preference utilities, and coherent risk measures.

## Findings and dispositions

| Adjacent surface | Existing meaning | #595 disposition |
|---|---|---|
| `voiage.methods.basic.evpi` and `DecisionAnalysis.evpi` | Risk-neutral expected net-benefit EVPI over sample-by-strategy values | Preserve unchanged. It is an affine-reduction oracle only when the #595 result verifies positive-affine utility and clairvoyant information. |
| Rust `voiage-numerics::evpi` | Stable monetary EVPI authority | Reuse only for differential assurance of the verified affine case; do not route nonlinear VoC through it. |
| `voiage.methods.preference` and `perspective` profile `utility_weights` | Metadata for already-constructed profile-specific net-benefit surfaces | Keep separate. These are preference/perspective heterogeneity inputs, not von Neumann–Morgenstern terminal-wealth utilities. |
| `voiage.financial.risk_analysis` VaR/CVaR | Tail-risk summaries of return samples | Keep separate. They neither optimize current/informed policies nor solve utility-equivalent information prices. |
| `voiage.multi_domain` finance utility | Domain helper using a negative-exponential transformation of a risk-adjusted return | Do not reuse as the #595 kernel. It lacks the versioned information structure, wealth/reference, price, policy, comparability and solver contracts. |
| `risk-sensitive/constrained-voi` issue #570 | Broader risk and feasibility family | Complementary planning only; passing #570 evidence cannot close #595. |
| `buying-price-voi` / maximum acquisition price | Named in issue prose, but no matching runtime, schema, fixture or registry family exists in the baseline | Record as a planned-name migration alias to #595 BPI. Do not imply prior implementation or compatibility. |

No existing runtime computes the required EUI, CEI, BPI, SPI or PPI family.
No existing fixture or result type can be relabeled as completion evidence.

## Canonical identity

- Method family: `expected-utility-information-pricing`.
- Canonical Python callable: `expected_utility_information_value`.
- Canonical result: one `ExpectedUtilityInformationResultV1` envelope.
- VoC presentation helper: `value_of_clairvoyance`, which delegates to the
  canonical callable with `information_kind="clairvoyant"` and returns the
  same result envelope plus presentation metadata.
- Planned-name alias: `buying-price-voi` resolves in registry/migration prose
  to the result's BPI measure; it is not a second callable.

The stable `evpi` name and return type remain unchanged. A nonlinear VoC result
must reject requests to present its value as raw monetary EVPI.

## Compatibility boundary

The new family is additive and experimental. It may be lazily exported from
`voiage.methods` and the package facade, added to DecisionAnalysis and CLI, and
registered under a new frontier fixture family. It must not:

- add required fields to stable v1 result schemas;
- change stable EVPI signatures or capability bits;
- import optional/JAX domain modules during base import;
- accept pre-transformed utility samples without a declared terminal-payoff
  and wealth convention; or
- advertise R, Julia or Mojo parity without installed shared-fixture evidence.

## VoC no-duplication checks

Tests and review must demonstrate that:

1. `value_of_clairvoyance` calls the canonical expected-utility pricing path;
2. the runtime contains no independent `voc` numerical function or Rust
   module;
3. the serialized result has a presentation label and selected-measure
   reference, not a second VoC scalar;
4. affine clairvoyance agrees with monetary EVPI; and
5. nonlinear utilities expose the explicit `affine_reduction_required`
   diagnostic instead of silently returning EVPI.
