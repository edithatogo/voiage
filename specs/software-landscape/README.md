# VOI/VOP software landscape

This directory is the versioned source for VOIAGE's software and method census.
It records observable capabilities rather than marketing parity.

- `registry.json` records search channels, software versions, licenses, sources,
  feature groups, and VOIAGE dispositions.
- `methods.json` distinguishes estimands, estimators, workflows,
  visualizations, applications, and related analyses.
- `method-evidence.json` and `method-evidence.schema.json` make the source and
  review state for every method family executable.
- `implementation-evidence.json` is generated from reviewed implementation and
  test paths and fails if a method is called `native` without executable
  repository evidence.
- `upstream-feature-evidence.json` and its schema record the reviewed upstream
  source, tests, documentation, examples, schemas, version, and limitations
  for every external feature without executing competitor code.
- `adjacent-method-dispositions.json` records why neighboring concepts are
  mapped to a canonical method, retained as related analyses/applications,
  admitted as a new method, or excluded from the VOI core.
- `schema.json` is the public software-registry contract.
- `gap-report.json` is generated from the registries and routes every
  non-equivalent feature or unresolved method-assurance gap to an existing
  Conductor implementation track.
- `parity-fixtures.json` prevents a `native` or `equivalent` external-feature
  claim unless independent, competitor-free fixtures and executable tests are
  named.
- The generated documentation matrix is produced by
  `scripts/generate_voi_feature_matrix.py`.

## Search and inclusion protocol

The quarterly and pre-minor-release search covers CRAN, R-universe, PyPI,
crates.io, Julia General, available Mojo channels, GitHub, GitLab, published
software reviews, public web tools, commercial documentation, and adjacent
Bayesian-design and active-learning ecosystems. Search terms include `value of
information`, `value of perspective`, `EVPI`, `EVPPI`, `EVSI`, `ENBS`,
`expected information gain`, `Bayesian experimental design`, `active learning`,
and `knowledge gradient`.

A record is included when authoritative package documentation, source,
registry metadata, or a peer-reviewed software description identifies a
relevant capability. A search hit alone is insufficient. Current versions and
licenses are pinned where an authoritative registry exposes them. Web and
commercial tools remain `not-reproducible` when current behavior cannot be
observed and pinned.

The census cannot prove universal completeness because registries and search
indexes are incomplete and terminology varies. A missed-candidate issue is a
normal correction path; it is not silently converted into a parity claim.

## Scientific classification

Every observed feature maps to a canonical method identifier. The registry
keeps distinct:

1. an estimand, such as EVPI or EVSI;
2. an estimator, such as nested Monte Carlo or moment matching;
3. a workflow or application;
4. a visualization;
5. a related analysis, such as an ICER or variance sensitivity index.

Information-theoretic acquisition is not represented as decision-theoretic or
economic VOI without an explicit action set and utility or loss.

## Canonical decision-problem boundary

`../core-api/schemas/v2/decision-problem.schema.json` is the versioned,
backend-neutral interchange contract used to determine whether an analysis is
decision-theoretic VOI. It requires alternatives, uncertain states, explicit
information actions and their costs, a utility/loss/net-benefit objective,
perspectives, population, time horizon, and provenance. Numerical draws and
model implementations remain referenced artifacts rather than being embedded
in the contract.

The v1 net-benefit-first schema remains supported for compatibility. Promotion
to v2 is additive: bindings may read v1, but stable cross-domain and
perspective-aware methods must emit or losslessly map to v2. An entropy,
acquisition, sensitivity, or prediction score without this decision boundary
remains a related analysis rather than economic VOI.
