# Normative v2.0 contract

`stable-api.json` is the machine-readable source of truth for the public v2.0
surface. It supersedes, but does not modify, the immutable v1.0 contract in
`../v1/`.

`extension-surface-policy.json` and `python-runtime-inventory.json` classify
the complete v2 Python adapter surface without changing the immutable v1
snapshots. Runtime inventory checks use these v2 manifests.

v2 makes the scientifically corrected expected value of sample information
(EVSI) boundary explicit:

- `normal_normal_two_arm_evsi` is a stable, Rust-owned analytical calculation
  for an equal-allocation, two-arm normal study.
- The `evsi` call signature is stable in v2, while estimator maturity remains
  method-specific. The built-in `method="two_loop"` route is fixture-backed
  when its prior, data-generating model, posterior update, and
  current-information calculation form one coherent Bayesian model.
- The built-in two-loop path fits one joint multivariate-normal prior from the
  probabilistic sensitivity analysis (PSA) sample. It uses that fitted prior
  for the current decision, directly sampled prior-predictive study data, and
  joint posterior calculations.
- Custom two-loop models provide both the trial simulator and joint posterior
  sampler as keyword-only callbacks.
- Simulation uses a local random generator. It never reads or mutates NumPy's
  process-global random state.
- The raw nested-Monte-Carlo point estimate is not silently truncated. A
  negative value remains visible so analysts can repeat seeds, increase loop
  counts, assess convergence, and check the coherence of custom study-model
  callbacks rather than treating it as exact zero.

These changes alter v1 numerical behaviour, callback availability, random
number generation, and the public signature. They therefore require the v2
major-version boundary documented in the migration guide.

The v2 compatibility policy retains semantic versioning and the stable-surface
rules from v1 for subsequent v2 releases. Runtime and binding conformance are
release gates; the presence of this contract alone does not claim that a v2
release has been published.
