# MoSCoW requirements — planned v1.2.0

## Must

- **M15-S1:** Return evaluated designs, feasible range/set and complete
  EVSI/cost/signed-ENBS curves.
- **M15-S2:** Return deterministic tie policy, selected optimum, boundary state
  and uncertainty around the optimum.
- **M15-S3:** Expose plotting inputs independently from the plotting library.
- **M15-S4:** Define EVSI/EVPI common-unit, zero-EVPI, finite and
  tolerance-aware bounds behavior.
- **M15-S5:** Allocate governed single-study optima by signed additive ENBS,
  permitting the empty portfolio and enforcing declared capacity, dependency,
  exclusion and guardrail constraints.
- **M15-S6:** Declare primary/secondary metrics, heterogeneous and delayed
  effects, interference, multiplicity, sequential monitoring, duration,
  stopping rules, opportunity cost, implementation delay and expected policy
  changes; return gross/net EVSI and ENBS without silently estimating omitted
  adjustments. Every model requires a provenance disposition of no-effect or
  prior COSS incorporation; added opportunity/delay costs require a
  provenance-backed exclusion from COSS research cost, and tolerance ties are
  anchored to the fixed global maximum.
- **M15-S7:** Distinguish a complete design enumeration from an evaluated-set-
  only result and label the optimum accordingly. Never infer optimality beyond
  the declared feasible set or evaluated designs.
- **M15-S8:** Add an explicit no-sampling comparator, economic-viability state
  and commissioning recommendation separately from the curve argmax and
  `best_evaluated_design`; all-negative ENBS must not recommend sampling.
- **M15-S9:** Make selection uncertainty replayable with a joint-replicate
  digest, seed/source, replicate count and unit, feasibility, tie policy,
  complete selection-probability mass, confidence level, confidence-set method
  and calibration status. Externally supplied probabilities remain labelled as
  externally supplied and do not imply calibrated assurance.

## Should

- Reconcile legacy clinical and plotting helpers and provide polyglot
  dispositions and independent argmax references.
- **M15-S10:** Propagate paired EVSI/EVPI uncertainty and suppress efficiency
  summaries for zero, weak or incompatible denominators.
- **M15-S11:** Report design-range expansion, boundary sensitivity, regret,
  near ties and winner's-curse diagnostics.

## Could

- Add domain-specific estimators for interference, multiplicity and delayed
  effects, plus optional experiment-platform adapters, behind the declared
  portfolio contract.

## Won't

- Extrapolate beyond the feasible set or relabel value/cost as EVSI/EVPI.
