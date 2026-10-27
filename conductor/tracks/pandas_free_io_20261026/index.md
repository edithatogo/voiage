# Track: Pandas-free external I/O and packaging

Status: in progress. T1.1/T1.2 and Phase 2 read semantics are implemented.
Native Excel, packaging reconciliation and pandas-free enforcement were accepted
in PRs #1205, #1206 and #1207. Phase 6 qualifies external producer compatibility
on Python 3.15.1 under the resource guard. T1.3 generated-code references,
Phase 7 repository cleanup and final whole-track acceptance remain open.
Historical receipts retain the state of their tested candidates.

GitHub issue: https://github.com/edithatogo/voiage/issues/1198 (parent: https://github.com/edithatogo/voiage/issues/1195)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)

Prerequisite: [Polars-native core containers](../polars_container_20261026/index.md), which gates the package-level acceptance. Successor: [translation reference](../translation_reference_20261026/index.md).
