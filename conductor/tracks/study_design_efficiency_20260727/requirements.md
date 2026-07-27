# MoSCoW requirements — planned v1.2.0

## Must

- **M15-S1:** Return evaluated designs, feasible range/set and complete
  EVSI/cost/signed-ENBS curves.
- **M15-S2:** Return deterministic tie policy, selected optimum, boundary state
  and uncertainty around the optimum.
- **M15-S3:** Expose plotting inputs independently from the plotting library.
- **M15-S4:** Define EVSI/EVPI common-unit, zero-EVPI, finite and
  tolerance-aware bounds behavior.

## Should

- Reconcile legacy clinical and plotting helpers and provide polyglot
  dispositions and independent argmax references.

## Could

- Extend the single-study result to constrained experiment portfolios.

## Won't

- Extrapolate beyond the feasible set or relabel value/cost as EVSI/EVPI.
