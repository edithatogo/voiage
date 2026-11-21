# Optional MLX/Metal EVPI backend

ML00 recorded a no-go for the tested Apple Silicon EVPI routes. Both
candidates missed the common-offset fallback p95 limit. Feasibility is
complete in this packet. Implementation is inapplicable.

- [ML00 decision and benchmark evidence](./evidence/ml00/README.md)
- [Follow-up memory profile](./evidence/ml00/memory-profile.md)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Scope and exploratory probes (PR #1277)](../../../docs/reviews/mlx-metal-evpi-scope-20261105.md)
- [Exploratory timing receipt](../../../docs/reviews/mlx-metal-evpi-timings-20261105.json)
- [Scientific ecosystem alignment](../scientific_ecosystem_alignment_20261029/index.md)
- [Archived Metal evidence contract](../../archive/apple-metal-production-speedup-evidence_20260625/index.md)
- [GitHub issue #1369](https://github.com/edithatogo/voiage/issues/1369)
