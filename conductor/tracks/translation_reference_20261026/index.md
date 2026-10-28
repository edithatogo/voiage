# Track: pandas and xarray to Polars translation reference

Status: completed. Implementation and final acceptance were authorised by the
maintainer and qualified on 2026-10-28.

GitHub issue: https://github.com/edithatogo/voiage/issues/1199 (parent: https://github.com/edithatogo/voiage/issues/1195)

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)

Prerequisite: [Polars-native core containers](../polars_container_20261026/index.md). Related: [pandas-free external I/O](../pandas_free_io_20261026/index.md).

Received [I/O acceptance and packaging handoff](./io-acceptance-handoff-python315.json)
from merged revision `420fd7ab`. The handoff retains its original planning snapshot; current authorization is
recorded in metadata.

[Executable translation reference](/voiage/user-guide/tabular-translation-reference)
is generated from `semantic-cases.json`. Source captures and retained consumer
notices are local, committed verification inputs; CI needs no external checkout.
All 15 local environments pass; see [qualification](./qualification-python315.json).
[Final acceptance](./acceptance-python315.json) records the passing hosted source
revision and documentation location. Current final-head CI is required before merge.
