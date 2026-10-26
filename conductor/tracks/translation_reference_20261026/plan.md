# Implementation plan: pandas and xarray to Polars translation reference

All tasks are pending. This track is registered planning only; implementation is not authorised. Prerequisite: `polars_container_20261026`.

## Phase 1: Derived inventory

- [ ] T1.1 — Freeze the derivation command over Voiage and the named consumer files; state the expected output shape before collecting idioms.
- [ ] T1.2 — Build the idiom inventory with file and line citations, marking anticipatory entries explicitly. (AC1).
- [ ] T1.3 — Validate: every entry cites real usage or is marked anticipatory; the derivation command and digest are recorded. (AC1).

## Phase 2: Semantic reference

- [ ] T2.1 — Draft the required rows: index absence, column access, row iteration, grouping and ordering, concatenation, alignment, reduction, rolling/resample boundaries, interpolation, null semantics and float formatting.
- [ ] T2.2 — Write the behavioural difference for each row, including the cases where Polars is not a drop-in replacement. (AC2).
- [ ] T2.3 — Validate: every required topic is present; no row asserts equivalence that Phase 3 does not test. (AC2).

## Phase 3: Executable parity

- [ ] T3.1 — Write the failing parity test for the first row before finalising its wording; repeat per row.
- [ ] T3.2 — Implement the test module, marking exact matches and bounded approximations separately. (AC3).
- [ ] T3.3 — Validate: every row has a test; the module passes in the pandas-free environment; rows requiring pandas are marked and skip cleanly when it is absent. (AC3).

## Phase 4: Non-equivalences

- [ ] T4.1 — Enumerate the operations with no equivalent and record the supported replacement, or state that none exists.
- [ ] T4.2 — Add the `%g` formatting, `attrs`, label-alignment and N-D interpolation entries. (AC4).
- [ ] T4.3 — Validate: each non-equivalence names a replacement or an explicit absence; none is presented as a silent gap. (AC4).

## Phase 5: Publication

- [ ] T5.1 — Place the reference in the documentation source and link it from the v2-to-v3 migration guide.
- [ ] T5.2 — Verify links, examples and the Astro build, and confirm the reference page and its pandas or xarray mentions sit on the `pandas_free_io_20261026` allow-list. (AC5).
- [ ] T5.3 — Validate: no example calls a removed symbol; the build passes; the migration guide links resolve. (AC5).

## Final acceptance

- [ ] CLOSE.1 — Review every AC against the exact tested revision; hosted checks must pass before any acceptance claim.
- [ ] CLOSE.2 — Reconcile plan, metadata and evidence; record the published location.
