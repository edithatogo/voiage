# Track: Pandas-free external I/O and packaging

Status: in progress. Maintainer authorised the recommended follow-up on 2026-10-27. T1.1 inventory and signature baseline are frozen. T1.2 CSV/Parquet/widget migration is locally qualified on Python 3.15.1/Polars 2.1.0; see [acceptance evidence](./t12-acceptance-python315.json). PR #1203 merged T1.2. Phase 2 CSV/Parquet inference, null/NaN propagation and serialization are locally qualified; see [read-semantics evidence](./phase-2-acceptance-python315.json). T1.3 and Phases 3–7, including native Excel and package-level acceptance, remain pending.

GitHub issue: https://github.com/edithatogo/voiage/issues/1198 (parent: https://github.com/edithatogo/voiage/issues/1195)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)

Prerequisite: [Polars-native core containers](../polars_container_20261026/index.md), which gates the package-level acceptance. Successor: [translation reference](../translation_reference_20261026/index.md).
