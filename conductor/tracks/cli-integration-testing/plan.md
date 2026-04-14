# Track Implementation Plan: CLI Completion and Integration Testing

## Phase 1: Complete CLI Commands [checkpoint: ]

### 1.1 Add missing EVSI/ENBS CLI commands [PENDING]
- [ ] Add `voiage calculate-evsi` command:
  - `--input-params` (CSV of PSA parameter samples)
  - `--input-nb` (CSV of net benefits, optional — can compute from model)
  - `--trial-design` (JSON specifying arm names and sample sizes)
  - `--method` (two_loop, regression, efficient, moment_based)
  - `--n-outer-loops`, `--n-inner-loops` (Monte Carlo parameters)
  - Output: EVSI value to stdout, optional JSON output
- [ ] Add `voiage calculate-enbs` command:
  - `--evsi` (EVSI value or file)
  - `--research-cost` (total cost of the proposed study)
  - Output: ENBS value to stdout

### 1.2 Add missing advanced method CLI commands [PENDING]
- [ ] Add `voiage calculate-adaptive-evsi` command:
  - Trial design JSON with adaptive design parameters
  - Interim analysis specifications
  - Output: Adaptive EVSI value
- [ ] Add `voiage calculate-portfolio-voi` command:
  - Portfolio JSON (list of candidate studies with costs)
  - Budget constraint
  - Method (greedy, integer_programming, dynamic_programming)
  - Output: Optimal portfolio and total VOI
- [ ] Add `voiage calculate-sequential-voi` command:
  - Dynamic spec JSON (time steps, population parameters)
  - Output: Sequential VOI at each time step

### 1.3 Add plotting CLI commands [PENDING]
- [ ] Add `voiage plot-ceac` command:
  - Input: net benefit data (CSV)
  - `--wtp-min`, `--wtp-max`, `--wtp-steps` (WTP range)
  - `--output` (save to file)
  - Output: CEAC plot (display or saved)
- [ ] Add `voiage plot-ceaf` command (once CEAF is implemented in Track 3)
- [ ] Add `voiage plot-voi-curves` command:
  - EVPI/EVPPI/EVSI curves against WTP or sample size
  - Output: plot (display or saved)
- [ ] Add `voiage plot-dominance` command (once dominance is implemented in Track 3)

### 1.4 Add structural EVPPI CLI command [PENDING]
- [ ] Add `voiage calculate-structural-evppi` command:
  - Mirror existing `calculate-structural-evpi` command
  - Support same JSON config format
  - Add `--parameters-of-interest` flag for partial structural EVPPI

### 1.5 Improve CLI developer experience [PENDING]
- [ ] Add `--format` option to all commands (text, json, csv)
- [ ] Add `--quiet` flag for scripting (suppress all output except result)
- [ ] Add `--verbose` flag for debugging (show intermediate calculations)
- [ ] Add example config generation: `voiage generate-config evsi > evsi_config.json`
- [ ] Ensure all `--help` output includes working examples

### 1.6 CLI e2e tests [PENDING]
- [ ] Write end-to-end tests for every CLI command using `typer.testing.CliRunner`
- [ ] Test happy path, invalid inputs, missing files, malformed configs
- [ ] Test `--format json` output is valid JSON
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90% for `voiage/cli.py`

---

## Phase 2: Enable All Integration Tests [checkpoint: ]

### 2.1 Remove test ignores incrementally [PENDING]
- [ ] **Pass 1:** Remove ignores for tests that should now pass (after Tracks 1-2):
  - `test_sample_information.py`
  - `test_plotting.py`
  - `test_structural.py`
  - `test_network_nma.py`
  - `test_network_meta_analysis.py`
  - `test_adaptive.py`
  - `test_portfolio.py`
  - `test_sequential.py`
  - `test_backends.py`
  - `test_analysis.py`
  - `test_analysis_comprehensive.py`
- [ ] Run each test file individually — fix failures before moving to next
- [ ] **Pass 2:** Remove ignores for remaining comprehensive tests
- [ ] **Pass 3:** Remove ignores for domain-specific tests (financial, environmental, healthcare)
- [ ] **Pass 4:** Remove ignores for CLI, coverage, and utility tests

### 2.2 Fix failing tests [PENDING]
- [ ] Categorize failures by root cause:
  - **Import errors:** Fix in Track 2 (should be resolved before this phase)
  - **Placeholder code failures:** Fix in Track 3 (should be resolved before this phase)
  - **Genuine test logic bugs:** Fix here — update test assertions, fix fixtures
  - **Flaky tests:** Add proper seeds, increase tolerances, or mark as flaky
- [ ] Fix tests in priority order: core method tests first, then integration, then domain-specific

### 2.3 Create missing integration tests [PENDING]
- [ ] `tests/test_full_workflow.py` — end-to-end: load data → run EVPI → run EVPPI → run EVSI → plot
- [ ] `tests/test_decision_analysis_methods.py` — test all DecisionAnalysis methods with realistic data
- [ ] `tests/test_backend_consistency.py` — verify NumPy and JAX backends produce consistent results
- [ ] `tests/test_cross_domain.py` — test healthcare, financial, environmental domain analyses
- [ ] `tests/test_cli_e2e.py` — comprehensive CLI end-to-end tests

### 2.4 Property-based and fuzzing tests [PENDING]
- [ ] Add hypothesis property-based tests for:
  - EVPI is always ≥ 0
  - EVPPI ≤ EVPI for any parameter subset
  - EVSI → 0 as sample size → 0
  - ENBS = EVSI - cost (identity)
  - Dominance is transitive
- [ ] Add fuzzing tests for CLI inputs (malformed JSON, empty files, extreme values)

---

## Phase 3: Performance and Regression Tests [checkpoint: ]

### 3.1 Add benchmark tests [PENDING]
- [ ] Add `tests/benchmarks/` directory with pytest-benchmark tests
- [ ] Benchmark EVPI performance at varying sample sizes (100, 1000, 10000, 100000)
- [ ] Benchmark EVPPI at varying parameter counts
- [ ] Benchmark EVSI (two-loop vs regression vs efficient) for speed and accuracy
- [ ] Benchmark JAX vs NumPy for each method
- [ ] Store baseline results for regression detection

### 3.2 Add regression test suite [PENDING]
- [ ] Save reference results to `tests/regression_data/` (known-good outputs for fixed inputs)
- [ ] Compare current outputs against reference with tolerance
- [ ] Run on every CI build to detect performance regressions

---

## Phase 4: Full Verification [checkpoint: ]

### 4.1 Run complete tooling suite [PENDING]
- [ ] `ruff check voiage/ tests/` — zero errors
- [ ] `mypy voiage/ --strict` — zero errors
- [ ] `pytest tests/ --cov=voiage --cov-fail-under=90` — all tests pass, coverage ≥90%

### 4.2 Verify no ignored test files remain [PENDING]
- [ ] Count `--ignore` patterns in pytest config — target: 0 (or at most a handful of justified exclusions)
- [ ] All test files should either pass or be genuinely deleted (not just ignored)

### 4.3 Verify CLI completeness [PENDING]
- [ ] Every method in `voiage.methods` has a corresponding CLI command
- [ ] Every plotting function has a CLI command
- [ ] `voiage --help` shows all commands organized by category

---

## Phase 5: Autonomous Track Review, Archive and Progression [checkpoint: ]

### 5.1 Phase Review Protocol — Execute after EVERY phase (1-4) [PENDING]
After completing each phase above, execute the following protocol **before** marking the phase `[x]` and proceeding:

1. **Commit phase changes:**
   - `git add -A`
   - `git commit -m "conductor(track4): Complete phase <N> of cli-integration-testing"`
2. **Invoke `/conductor:review`** targeting all changes since the previous checkpoint commit.
3. **Apply all Critical and High severity fixes** identified by the review automatically.
4. **Re-run verification:** `ruff check voiage/ tests/ && mypy voiage/ --strict && pytest tests/ --cov=voiage --cov-fail-under=90 -q`
5. **If failures persist after 2 fix attempts:** Halt and report to user with details.
6. **Commit review fixes:** `git add -A && git commit -m "fix(conductor): Apply automated review fixes for phase <N>"`
7. **Mark phase complete** in this plan file (change `[PENDING]` → `[x]`).

### 5.2 Track Completion Protocol — Execute after final phase [PENDING]
After Phase 4 is complete and all phase reviews pass:

1. **Invoke `/conductor:review`** targeting the **entire track** (from track start commit to HEAD).
2. **Apply all Critical, High, and Medium severity fixes** automatically.
3. **Re-run full test suite:** `ruff check voiage/ tests/ && mypy voiage/ --strict && pytest tests/ --cov=voiage --cov-fail-under=90`
4. **Commit review fixes:** `git add -A && git commit -m "fix(conductor): Apply final track review fixes for cli-integration-testing"`
5. **Archive the track:**
   - `mkdir -p conductor/archive && mv conductor/tracks/cli-integration-testing conductor/archive/cli-integration-testing`
   - Update `conductor/tracks.md`: change `[ ]` → `[x]` for this track, add `[completed: <date>]` and archive link.
6. **Commit archive:** `git add -A conductor/ && git commit -m "chore(conductor): Archive completed track cli-integration-testing"`
7. **Read next track:** Read `conductor/tracks/docs-developer-experience/plan.md` and begin execution from Phase 1, Task 1.1.
8. **Announce:** "Track 4 (cli-integration-testing) complete. Starting Track 5 (docs-developer-experience)."
