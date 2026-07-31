# Risk-Adjusted Information Pricing and VoC Presentation

## Overview

Implement GitHub issue [#595](https://github.com/edithatogo/voiage/issues/595)
as one governed expected-utility information-pricing family. Expected Value of
Clairvoyance (VoC) is a presentation of the same clairvoyant-policy result,
never a duplicate numerical method or an unconditional alias for monetary
EVPI.

Owning issue: [#595](https://github.com/edithatogo/voiage/issues/595), native
sub-issue of [#318](https://github.com/edithatogo/voiage/issues/318) under
programme [#313](https://github.com/edithatogo/voiage/issues/313). Planned
release v1.2.0, MoSCoW Must, canonical C16 requirements M16 and M17.

## Requirements

1. Represent a versioned terminal-payoff decision problem with state
   probabilities, uniquely named strategies, initial wealth/reference state,
   payoff units, stakeholder/organizational scope, information structure,
   cost location, and deterministic tie policy.
2. Support named, serializable, strictly increasing utility descriptors rather
   than arbitrary callables. The first contract supports affine, exponential
   (CARA), logarithmic, and power (CRRA) utilities with explicit parameters,
   normalization, domains, and inverse utilities.
3. Calculate the current-policy and clairvoyant-policy expected utilities,
   expected utility increase (EUI), certainty-equivalent increase (CEI),
   buying price (BPI), selling price (SPI), and probability price (PPI) against
   a declared terminal-outcome floor. Preserve signed results and never
   silently clamp a negative value or a failed root.
4. Define BPI as the price paid for the information that makes the optimized
   clairvoyant-policy lottery indifferent to the uninformed current-policy
   lottery. Define SPI as compensation for surrendering information that makes
   the optimized uninformed-policy lottery indifferent to the unpriced
   clairvoyant-policy lottery. Re-optimize the applicable policy at each trial
   price and record cost/compensation location.
5. Return policy switches, willingness-to-pay boundaries, deterministic
   root-finding brackets, residuals, iterations, convergence states, and
   actionable failure diagnostics.
6. Define PPI from a declared lower outcome anchor `z0` by the mixture
   indifference relation `PPI * U(z0) + (1 - PPI) * I(0) = B(0)`. Validate the
   anchor, direction, normalization and `[0, 1]` bounds. Other constructed-scale
   prices are unsupported until they declare equivalent anchors and transfer
   semantics.
7. Return explicit comparability conditions. EUI magnitude is utility-scale
   dependent; prices require commensurate monetary/payoff units and the same
   declared utility/wealth convention. Cross-problem EUI/BPI rankings agree
   only under the supported theoretical conditions; CEI/BPI agreement for
   affine or exponential utility must be stated rather than generalized.
8. Present `voc` from the canonical expected-utility/clairvoyant-policy result.
   Monetary EVPI equality may be asserted only for a declared positive-affine
   utility reduction. Nonlinear utility must retain the distinction.
9. Reconcile adjacent risk-sensitive, CVaR, maximum-acquisition-price,
   preference, and buying-price dispositions without treating them as #595
   implementation evidence.
10. Provide deterministic schemas, normative fixtures, Rust and Python runtime
   surfaces, CLI/reporting/docs, and explicit R, Julia, and Mojo dispositions.

## Acceptance criteria

- **AC-01:** The issue hierarchy, Project 28, metadata, registry, roadmap,
  MoSCoW M16/M17, Mermaid design, canonical C16 projection and cross-reference
  manifest agree on this focused track.
- **AC-02:** Utility, wealth/reference state, probabilities, units, cost
  location, information structure, policies, scope, tie policy, and estimator
  provenance are explicit and deterministically serializable.
- **AC-03:** EUI, CEI, BPI, SPI and PPI implement the definitions above with signed
  values, policy re-optimization, versioned results, and complete root
  diagnostics.
- **AC-04:** VoC delegates to or presents the canonical result; no second
  numerical kernel exists and raw EVPI is used only for a verified affine
  reduction.
- **AC-05:** Independent affine and enumerable references, nonlinear
  counterexamples, buy/sell asymmetry, positive-affine invariance, utility
  domain failures, root-bracketing failures, policy ties, nonuniform
  probabilities and multi-stakeholder comparability cases precede positive
  claims.
- **AC-06:** Rust owns accepted numerical policy; Python is a typed thin
  facade. R, Julia and Mojo are either fixture-verified or explicitly
  unsupported/not verified.
- **AC-07:** Documentation and capability discovery distinguish expected
  utility, monetary value, certainty equivalents, preference heterogeneity,
  coherent risk measures, and acquisition-price helpers.
- **AC-08:** Focused tests, property/differential assurance, serialization,
  coverage, repository harness, full Conductor validation and hosted required
  checks pass before repository completion.

## Non-functional constraints

- Preserve the released EVPI API and stable v1 wire contracts.
- Use finite validation, deterministic first-index ties, bounded numerical
  work, and fail-closed utility domains and root solving.
- Keep the family experimental or fixture-backed until scientific promotion
  and installed cross-language evidence justify a stronger maturity label.
- Do not introduce an arbitrary callable utility into serialized or polyglot
  contracts.

## External and human gates

- Scientific review is required before stable promotion.
- Hosted checks, merge, release, publication, registry acceptance, and issue
  closure are separate gates.
- Organizational utility and stakeholder aggregation choices remain
  user-supplied assumptions, not inferred facts.

## Out of scope

- A second VoC kernel, a blanket alias from VoC to monetary EVPI, or
  reinterpretation of preference heterogeneity as risk attitude.
- Automatic aggregation of incompatible stakeholder utilities.
- Claims that EUI is cardinally comparable after arbitrary positive-affine
  utility rescaling.
- Unbounded optimization, stochastic root solvers, or silent extrapolation
  beyond a declared bracket.

## Authoritative inputs

- GitHub issue #595, revision observed `2026-07-27T08:29:39Z`.
- Repository baseline `7b9e50f2dc52205311bc8647bee06fe10c19f22a`.
- Abbas and Hazen, *On the Value of Information Across Decision Problems*,
  DOI `10.1287/deca.2024.0187` (publisher metadata and abstract checked
  2026-07-31).
- Hazen, Borgonovo and Lu, *Information Density in Decision Analysis*, DOI
  `10.1287/deca.2022.0465` (publisher metadata and abstract checked
  2026-07-31); used only for adjacent information-scale context, not as the
  sole authority for price definitions.
- Hazen and Sounderpandian, *Value of Information in Decision Analysis: The
  Case of EUI, CEI, and SPI*, 1999, used for CEI/SPI/PPI distinctions and
  reviewed through the author-paper record cited by the scientific audit.
- `conductor/requirements.md`, `conductor/design.md`,
  `conductor/tracks/supported_frontier_method_completion_20260723/`,
  `specs/core-api/extension-evolution.md`, and the Rust-authority policy in
  `conductor/product-guidelines.md`.
