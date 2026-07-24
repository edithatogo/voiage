# Stable Rust Core Pre-start Audit

## Boundary

This audit prepares issue #316 without starting the track or changing the
scientific contract. Formal implementation remains blocked on accountable
human approval of scientific-freeze candidate
`9f437ea0b0521297b81f66adfac980e537db3c0ebf63823445f3bff2d285c3f9`.

## Confirmed remaining stable-authority gap

The generated implementation registry identifies `net-benefit` as the only
stable method with `authority_state: python-stable-gap`. Expected opportunity
loss, EVPI, EVPPI, EVSI, ENBS, CEAF, dominance, heterogeneity, and structural
aggregation already have Rust numerical kernels. This is an implementation
classification, not a claim that all estimator-assurance and binding work in
issue #316 is complete.

## Current Python surfaces

1. `voiage.core.utils.calculate_net_benefit` is the exported array helper. It
   computes `effects * willingness_to_pay - costs` for one- or two-dimensional
   equal-shaped cost and effect arrays.
2. `voiage.health_economics.calculate_net_monetary_benefit_simple` separately
   computes the scalar form and would remain a silent divergent stable path if
   only the array helper were routed through Rust.
3. `voiage.core.__init__` preserves the public array-helper alias.
4. `tests/test_utils_comprehensive.py` is the current behavioral baseline but
   does not test non-finite inputs, overflow, Rust/Python equivalence, or a
   stable serialized result envelope.

## WTP shape ambiguity

The current helper has four observable cases:

- scalar WTP returns the cost/effect shape;
- a threshold vector appends a threshold axis;
- a two-dimensional WTP array with two-dimensional inputs uses ordinary NumPy
  broadcasting despite documentation suggesting sample-by-threshold semantics;
- otherwise it falls back to ordinary NumPy broadcasting.

The third and fourth cases are not a sufficiently precise polyglot contract.
The recommended additive v1.1 policy is:

1. scalar WTP: preserve the input shape;
2. threshold vector: append a threshold axis;
3. sample-by-threshold WTP: require the first dimension to equal the sample
   dimension and return sample-by-strategy-by-threshold values;
4. reject other WTP shapes with a typed dimension diagnostic;
5. retain finite negative WTP values because generalized-benefit objectives
   may use them, while documenting their interpretation;
6. reject non-finite costs, effects, WTP values, and non-finite arithmetic
   results.

The existing element-wise two-dimensional behavior can remain available only
through an explicit compatibility path during a documented deprecation
window. It must not be silently inferred from shape.

## Required implementation route

1. Add Rust input/output types that preserve sample, strategy, and threshold
   dimensions without relying on language-specific broadcasting.
2. Add analytical, dimension, non-finite, overflow, negative-WTP, axis,
   permutation, scaling, and property tests before the kernel.
3. Add the Rust kernel and public `voiage-numerics` facade export.
4. Add a PyO3 `compute_net_benefit` boundary and lazy Python runtime adapter.
5. Route both public Python calculation helpers through the native boundary;
   retain only explicit, tested compatibility behavior.
6. Add versioned fixtures/result schema or a documented array-return contract,
   Rust/Python differential tests, benchmarks, mutation coverage, and
   capability-registry evidence.
7. Regenerate implementation evidence, the gap report, feature matrix, and
   scientific candidate after implementation. Any candidate-digest change
   invalidates a prior approval and requires a new review.

## Non-goals

This audit does not approve the freeze, start issue #316, promote a method,
waive manual verification, or satisfy Rust, Python, ABI, binding, release, or
external gates.
