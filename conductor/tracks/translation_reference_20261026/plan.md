# Implementation plan: pandas and xarray to Polars translation reference

Implementation authorised by the maintainer on 2026-10-28. Implementation is locally qualified; the accepted I/O handoff is retained unchanged. Hosted implementation checks passed; final records-only delivery remains check-gated. Prerequisite: `polars_container_20261026`.

## Phase 1: Derived inventory [checkpoint: ea646644]

- [x] T1.1 — Freeze the derivation command over Voiage and capture the `tobac` consumer idioms (`scripts/run_empirical_longitudinal.py`, `scripts/run_release_calibration.py`, `scripts/verify_empirical_longitudinal.py`) into a committed, digest-pinned JSON manifest (`tobac-consumer-idioms.json`) so CI never requires an external checkout; state the expected output shape before collecting idioms.
- [x] T1.2 — Build the idiom inventory with file and line citations (against Voiage and `tobac-consumer-idioms.json`), marking anticipatory entries explicitly. (AC1).
- [x] T1.3 — Validate: every entry cites real usage in Voiage or `tobac-consumer-idioms.json`, or is marked anticipatory; the derivation command and digests are recorded and hermetic. (AC1).

## Phase 2: Semantic reference [checkpoint: ea646644]

- [x] T2.1 — Draft the required rows: index absence, column access, row iteration, grouping and ordering, concatenation, alignment, reduction, rolling/resample boundaries, interpolation, null semantics and float formatting.
- [x] T2.2 — Write the behavioural difference for each row, including the cases where Polars is not a drop-in replacement. (AC2).
- [x] T2.3 — Validate: every required topic is present; no row asserts equivalence that Phase 3 does not test. (AC2).

## Phase 3: Executable parity [checkpoint: ea646644]

- [x] T3.1 — Write the failing parity test for the first row before finalising its wording; repeat per row.
- [x] T3.2 — Implement the test module, marking exact matches and bounded approximations separately. (AC3).
- [x] T3.3 — Validate: every row has a test; the module passes in the pandas-free environment; rows requiring pandas are marked and skip cleanly when it is absent. (AC3).

## Phase 4: Non-equivalences [checkpoint: ea646644]

- [x] T4.1 — Enumerate the operations with no equivalent and record the supported replacement, or state that none exists.
- [x] T4.2 — Add the `%g` formatting, `attrs`, label-alignment and N-D interpolation entries. (AC4).
- [x] T4.3 — Validate: each non-equivalence names a replacement or an explicit absence; none is presented as a silent gap. (AC4).

## Phase 5: Publication [checkpoint: ea646644]

- [x] T5.1 — Place the reference in the documentation source and link it from the v2-to-v3 migration guide.
- [x] T5.2 — Verify links, examples and the Astro build, and confirm the reference page and its pandas or xarray mentions sit on the `pandas_free_io_20261026` allow-list. (AC5).
- [x] T5.3 — Validate: no example calls a removed symbol; the build passes; the migration guide links resolve. (AC5).

## Final acceptance [checkpoint: ffa05870]

- [x] CLOSE.1 — Review every AC against the exact tested revision; hosted checks must pass before any acceptance claim.
- [x] CLOSE.2 — Reconcile plan, metadata and evidence; record the published location.

Local evidence: `capture-contract.json`, both immutable source inventories,
`semantic-cases.json`, and the generated reference page. All 14 target recipes
and all 14 external legacy forms passed on Python 3.15.1 under the resource
guard. All 15 local environments passed at `ea646644`; see `qualification-python315.json`. Hosted checks passed at `3925ba1f`; see `acceptance-python315.json`.

Implementation/review checkpoint: `ea646644484112d47504a81e478064e065bd6e64`.
The initial MDX and link failures are retained in the qualification receipt;
the final candidate passes its complete local gate. No runtime inputs changed.

Final acceptance: AC1–AC5 reviewed against the exact tested implementation;
all applicable hosted checks passed at `3925ba1f170533d63815618b6f0a2055fc23ebd1`.
The documentation source and built route are recorded in `acceptance-python315.json`.
Current final-head CI remains mandatory before merge. Track paths remain fixed
for the executable inventory, citation and reference permissions.
