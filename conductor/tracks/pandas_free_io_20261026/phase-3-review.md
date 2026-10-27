# Review Report: Pandas-free I/O Phase 3

## Summary
The native Excel reader and writer preserve the frozen worksheet behavior on Python 3.15.1. Hosted acceptance remains required on the PR revision.

## Verification Checks
- [x] **Plan Compliance**: T3.1-T3.3 implemented and locally qualified. The named AC3 raw-cell reader alternative was documented before implementation after default-reader compatibility failures.
- [x] **Style Compliance**: Ruff, typecheck, repository/security harness, prose and documentation gates passed.
- [x] **New Tests**: Frozen producer/oracle workbooks cover mixed scalars, typed/duplicate headers, blank areas, chart sheets, dates/time, missing/error cells, multi-sheet export, formula caches, literal URLs, non-finites, empty tables, unavailable engines and cleanup on failures. A subprocess blocks pandas, openpyxl and fastexcel.
- [x] **Test Coverage**: 94.72% aggregate; critical-module policy passes; all 93 changed executable lines and 41 changed branches covered.
- [x] **Test Results**: Twelve tox environments passed outright. Full suite: 5,286 passed, 14 capability skips, one stale engine-extra assertion failed. The profile/test were corrected, and its failure-only rerun passed in the same coverage environment. No runtime, dependency or shared-fixture changes followed the full gate; unchanged evidence is reused.

The implementing agent reviewed each changed source, test, dependency and CI file. No unresolved Critical, High, Medium or Low findings remain. Both engines have actual Python 3.15.1 installation/import/workbook evidence and recorded upstream licences. CI exercises binary installation and frozen behavior on Linux, macOS and Windows.

Calamine and XlsxWriter use context-managed workbooks. A guarded 120-cycle probe found stable descriptors after initialization, no retained workbook file descriptors and approximately 0.1 MiB post-warmup RSS growth. The resource guard bounds sampled combined RSS, elapsed time and free space; this bounded observation is not a universal leak-free guarantee.

All 47 existing callable signatures match the merged prerequisite. Public payload keys, tuple shapes, typed headers, warning/empty-result behavior and workbook cell semantics remain. Numerical kernels and wire/binding contracts are unchanged. Core dependency removal and generated notebook templates remain later phases. See `phase-3-acceptance-python315.json` for exact revisions, raw initial failures, repaired evidence, hashes and remaining hosted gates.

## Review addendum: whitespace compatibility

PR review subsequently identified whitespace-only row loss. Native Calamine also
trimmed text when a producer omitted XML-space preservation. The compatibility
pass now streams secure SAX output into an owned disk-backed archive before
native reading, preserving source bytes and temporary-file cleanup. Fifteen
whitespace/padded-text combinations compare against the old producer reader;
additional tests reject DTD input and verify cleanup and source immutability.

The new full suite passed 5,304 tests with 14 capability skips. The 13 local
environments are qualified by that full gate plus successful reruns of lint
and typecheck after annotation-only and documented output-handler false-positive
repairs; executable function bodies/line positions are unchanged. Coverage is
94.73%, with all 127 changed executable lines and 49 changed branches covered.
A clean wheel install with only the Excel extra passed without either legacy
engine installed, with pandas/openpyxl/fastexcel imports blocked. A fresh guarded
120-cycle probe retained no workbook descriptors and showed no post-warmup RSS
growth. Updated hosted checks remain mandatory. See
`phase-3-whitespace-python315.json`; the earlier receipt is historical evidence.
