# Ecosystem Fixture Expectations

This directory reserves fixture expectations for HEOR ecosystem-level
integrations that connect `voiage` to sibling projects without importing their
internals.

## Planned Fixture Families

- `lifecourse`: deterministic health-economic run bundles with VOI-ready
  net-benefit and parameter-sample artifacts.
- `innovate`: health-intervention adoption/diffusion uncertainty artifacts that
  can drive VOI workflows where implementation uptake is itself uncertain.
- `mars`: fixed-API surrogate/metamodel fixtures used to validate optional
  regression-style VOI backends.

## Fixture Policy

- Fixtures must be deterministic and small enough for CI.
- Portable interchange formats should be JSON metadata plus CSV, Arrow, or
  Parquet tables as needed.
- Shared fixtures must not depend on pickle.
- Each fixture family should come with explicit versioning, provenance, and
  expected outputs before it can be marked supported.
