# Review Report: Pandas-free I/O Phase 6

## Summary
Existing producer interoperability is qualified through a real installed-wheel
lane without changing the public API, runtime, numerical or wire behavior.

## Verification Checks
- [x] **Plan Compliance**: T6.1-T6.3 locally qualified; hosted acceptance remains a delivery gate.
- [x] **Style Compliance**: Required lint, type, prose, harness and documentation gates pass.
- [x] **New Tests**: 21 producer controls and two automation contracts. Red wiring contracts precede activation; actual negative import requests fail. No new skip or retry.
- [x] **Test Coverage**: Aggregate and critical-module policies pass at 94.73%; no changed runtime lines exist.
- [x] **Test Results**: All 15 tox environments pass; full suite 5,361 passed, 14 existing capability skips. Dedicated installed-wheel lane passes all 21 cases.

The implementing agent reviewed AC6/AC7, frozen fixtures, real buffer identity,
protocol/copy distinctions, import ownership and restoring the test import hooks,
fresh-process/package-prefix assertions, tox and protected/platform CI wiring.
No Critical, High, Medium or Low implementation findings remain. Standard
producer tooling is explicit and unpublished; no production shim is introduced.
The separate no-extras absence environment is retained. The producer process
already contains caller-owned pandas, so its proof concerns import ownership,
not a false claim that pandas is absent from that process.

The actual numeric Polars buffers are shared on the capsule path. This does not
change the existing copy diagnostic or guarantee sharing for every dtype.
Strict-copy rejections remain stable; Arrow Table strict-copy identity passes.
Rows, binding values, nullable/category/timezone fields, empty schemas, slices,
content/schema/manifest hashes and nonconforming-object diagnostics are tested.
The runtime, Rust, binding and pinned wire trees remain byte-identical to the
merged prerequisite. There is no new public persistence format.

The final candidate was tested under Python 3.15.1 with a 4 GiB sampled group
limit, two coverage workers and one native thread. Peak RSS was 1,787,904 KiB;
the successful temporary tree was removed. Exact inputs, report hashes and
remaining hosted/Phase 7/Conda boundaries are retained in
`phase-6-acceptance-python315.json`. This is a phase review, not whole-track
completion or release approval.
