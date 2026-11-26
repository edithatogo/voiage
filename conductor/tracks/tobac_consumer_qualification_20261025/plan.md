# Implementation plan — tobac consumer qualification

The original planning-only hold was superseded by the user's 29 October
direction to implement, review and archive this track. The [installed M2 retail
workflow](evidence/installed-m2-retail-20261124/README.md) passed on 24 November.
Earlier candidate results are in the [November qualification packet](evidence/installed-20261101/README.md).
[Specification](spec.md). [Repository workflow](../../workflow.md).

## Phase 1 — Consumer reconciliation

- [x] Task P1.1: Refresh producer/master revisions, ownership and current contracts.
  The [current master record](https://github.com/edithatogo/tobac/tree/fe06ef5958ba0f218317a738fe8de95973300a12/conductor/archive/scientific_ecosystem_20261025)
  retains tobac ownership of the domain model. Its first Voiage use is the
  frozen M2 retail case through the published 3.1.0 optional backend.
- [x] Task P1.2: Map consumer requirements to existing behavior and evidenced gaps.
- [x] Task P1.3: Write failing consumer and independent-oracle tests before changes.
  - [x] Map master R11.1–R11.5 to existing parameter/model/execution contracts; write round-trip, invalid-input, scenario-isolation, input-mutation and I/O-independent calculation cases.
    Covered by `tests/test_tobac_r11_contracts.py` alongside `tests/test_tobac_parameter_artifact.py`, `tests/test_tobac_structure_only.py`, `tests/test_tobac_shared_cost_boundary.py`, and `tests/test_tobac_shared_cost_v2.py` against the installed Python 3.15.2 wheel.
- [x] Task P1.4: Phase verification and checkpoint under the repository workflow.
  Verified installed-wheel `tobac` contract suites and Conductor governance checks on 26 November 2026.

## Phase 2 — Bounded producer support

- [x] Task P2.1: Qualify required Python/native dependencies and record genuine blockers.
  Verified `.venv` Python `3.15.2`, `voiage 3.1.0` (`_core.abi3-darwin.so`), `rustc 1.100.0`, `check_pandas_free.py`, and locked `voiage-domain`/`voiage-numerics` unit tests (29 passed); recorded genuine blockers (`deep_learning` PyTorch CPython 3.15 gap, `information_source_portfolio` `probability > 0` schema, offline PyPI SSL timeout in `dependency_frontier.py`, and 3.1.0 vs post-3.1.0 `main` adapter scope) in [Phase 2 receipt](evidence/phase2-qualification-20261126/receipt.json).
- [x] Task P2.2: Implement only demonstrated reusable gaps within existing core/API ownership.
  - [x] Close only demonstrated parameter/model/execution separation gaps through existing typed/native APIs, preserving structural invariants and domain ownership.
    Phase 1 R11.1–R11.5 (`tests/test_tobac_r11_contracts.py`) and existing `tobac` suites verified `AnalysisSpec`, `ParameterSpec`, `NumericalPolicy`, `RunContext`, `NormalizedInputBundle`, `run_evpi`, `information_source_portfolio_value`, and `paired_draw_bundle` with zero runtime defects; no new runtime code or wrapper API added.
- [x] Task P2.3: Qualify optional backend/interchange paths against exact references and measured value.
  Qualified Polars/Arrow zero-copy `paired_draw_bundle` + IPC/Parquet round-trip, `ValueArray.from_numpy` vs `ValueArray.from_polars` parity, `evpi`/`run_evpi`/`JaxBackend` (`x64`) parity (`<= 4.00e-15` across `4,096 x 4` and `20,480 x 4`, exact `Fraction(1, 2**34)` oracle), and median-of-5 timings in `tests/test_tobac_phase2_interchange_and_backends.py` and [Phase 2 evidence](evidence/phase2-qualification-20261126/README.md).
- [x] Task P2.4: Phase verification and checkpoint under the repository workflow.
  Verified `ruff check`/`ruff format --check` and `.venv/bin/python -I scripts/run_installed_pytest.py` across `tests/test_tobac_phase2_interchange_and_backends.py` (4 passed) and `tests/test_tobac_r11_contracts.py` (6 passed) on 26 November 2026.

## Phase 3 — Installed consumer and evidence

- [x] Task P3.1: Build and install independent artifacts; verify actual import/artifact identities.
- [x] Task P3.2: Execute consumer contracts, negative cases, precision and scientific invariants.
  - [x] Qualify the [installed frozen M2 retail EVPI workflow](evidence/installed-m2-retail-20261124/README.md)
    with both wheels, the frozen reference, reversed state order and error controls.
  - [x] Prove a bounded extension preserves reference scenarios and record resolved parameters, provenance, model/schema identity and execution settings separately (R11.1–R11.5).
    Verified with non-editable `tobac_regulatory_research==0.2.0` (`fe06ef5958`) and `voiage==3.1.0` (`d0cd33b45`) wheels under `python -I`: `test_voiage_installed.py` (3 passed) + `test_ecosystem_contracts.py` (12 passed), plus `crime_weight=1.5` bounded extension and independent R11.1–R11.5 provenance layers in [Phase 3 R11 evidence](evidence/installed-r11-extension-20261126/README.md).
- [x] Task P3.3: Review existing engineering/context/security controls; add only justified gaps.
  Verified `repo_harness.py` (43 workflows, 0 findings), 60 control/contract pytest checks (`test_repo_harness.py`, `test_execution_packets.py`, `test_version_sync.py`, `test_mutation_score.py`), `check_pandas_free.py`, `check_reference_policy.py` (5,012 files), `voiage/versioning.py`, and `ruff --select=S --ignore=S101`; documented `pytest-gremlins`/`mutmut` assessment with zero runtime additions in [controls review](evidence/phase3-controls-review-20261126/README.md).
- [x] Task P3.4: Phase verification and checkpoint under the repository workflow.
  Verified Phase 3 installed consumer suites, controls review, and Conductor governance checks on 26 November 2026.

## Phase 4 — Handoff

- [~] Task P4.1: Execute full relevant native, clean-package, coverage, security and hosted gates.
- [~] Task P4.2: Reconcile main and master track, publish only within authority, verify receiving refs/files.
- [x] Task P4.3: Deliver exact artifact/API/schema receipts and rollback notes to tobac.
  Recorded in [handoff-20261126.md](evidence/handoff-20261126.md).
- [ ] Task P4.4: Phase verification and checkpoint including repository review requirements.

Current release evidence and available-compute boundary are recorded in
[the 2 November reconciliation](../scientific_ecosystem_alignment_20261029/reconciliation-20261102.md).
Existing unchecked native acceptance tasks remain open.
