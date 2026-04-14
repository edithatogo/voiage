# Track Implementation Plan: Replace Placeholders and Implement Missing Methods

## Phase 1: Replace Placeholder Implementations [checkpoint: ]

### 1.1 Fix Sequential VOI `_calculate_evpi_at_step()` [PENDING]
- [ ] **Current state:** Returns `total_variance * abs(wtp) * 0.01` — arbitrary scaling, not real EVPI
- [ ] **Required:** Implement actual EVPI calculation at each sequential decision step
- [ ] Research the correct algorithm: EVPI at step t = E[max_i NB_i(θ)] - max_i E[NB_i(θ)] given information available at step t
- [ ] Replace placeholder in `voiage/methods/sequential.py` line ~189
- [ ] Write unit tests verifying mathematical properties:
  - EVPI ≥ 0 at every step
  - EVPI decreases (or stays flat) as more information is gathered
  - EVPI → 0 when all uncertainty is resolved
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90% for `sequential.py`

### 1.2 Fix JAX EVSI `_evsi_two_loop_jax_core()` [PENDING]
- [ ] **Current state:** Returns `0.0` placeholder, falls back to NumPy (`main_backends.py` line ~325)
- [ ] **Required:** Implement actual JAX-optimized two-loop EVSI
- [ ] Implement JAX-compatible Bayesian update for Normal-Normal conjugate model
- [ ] Use `jax.vmap` for outer loop parallelization
- [ ] Use `jax.lax.scan` or `jax.vmap` for inner loop parallelization
- [ ] Benchmark: JAX EVSI should be at least 2x faster than NumPy for n_samples ≥ 1000
- [ ] Write unit tests verifying JAX and NumPy results agree within tolerance
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90% for JAX backend EVSI code

### 1.3 Fix Observational VOI built-in modeler [PENDING]
- [ ] **Current state:** Requires user-provided `obs_study_modeler` callback — no built-in implementation
- [ ] **Required:** Provide at least one built-in observational study modeler
- [ ] Implement a basic Bayesian observational study modeler:
  - Propensity score matching/weighting model
  - Outcome regression model
  - Bayesian posterior sampling for counterfactual outcomes
- [ ] Default modeler should work with standard observational study designs (cohort, case-control)
- [ ] Write unit tests with synthetic observational data
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90% for `observational.py`

### 1.4 Fix Calibration VOI built-in modeler [PENDING]
- [ ] **Current state:** Requires user-provided `cal_study_modeler` callback — no built-in implementation
- [ ] **Required:** Provide at least one built-in calibration study modeler
- [ ] Implement a basic Bayesian calibration modeler:
  - Prior distribution over model parameters
  - Likelihood function for calibration data
  - MCMC sampling via NumPyro for posterior
  - VOI calculation from posterior samples
- [ ] Write unit tests with synthetic calibration data
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90% for `calibration.py`

---

## Phase 2: Implement Efficient/Moment-Based EVSI [PENDING] [checkpoint: ]

### 2.1 Research and design [PENDING]
- [ ] Study Heath et al. (2018) "Estimating the Expected Value of Sample Information Using the Probabilistic Sensitivity Analysis Sample" method
- [ ] Study the moment-based method: EVSI ≈ 0.5 * (∂²ENB/∂θ²) * Var(θ|data) * n/(n+n₀)
- [ ] Determine which method(s) to implement — recommend both for completeness
- [ ] Design API: `evsi(..., method="efficient")` and/or `evsi(..., method="moment_based")`
- [ ] Document mathematical formulas and assumptions

### 2.2 Implement efficient EVSI [PENDING]
- [ ] Create `voiage/methods/efficient_evsi.py` or add to `sample_information.py`
- [ ] Implement the regression-based efficient method:
  - Fit metamodel to PSA net benefits
  - Use posterior variance reduction to estimate EVSI without inner loop
  - Support multiple metamodel types (GAM, RF, GP if available)
- [ ] Implement the moment-based method:
  - Calculate expected net benefit gradient and Hessian
  - Apply Fisher information-based variance reduction formula
  - Support for multi-parameter settings
- [ ] Write comprehensive unit tests
- [ ] Benchmark against two-loop MC: should be 10-100x faster with comparable accuracy
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90%

### 2.3 Add CLI command for efficient EVSI [PENDING]
- [ ] Add `voiage calculate-evsi` CLI command with `--method` option (two_loop, regression, efficient, moment_based)
- [ ] Document usage in CLI help

---

## Phase 3: Implement CEAF [PENDING] [checkpoint: ]

### 3.1 Research and design [PENDING]
- [ ] Study the Cost-Effectiveness Acceptability Frontier (CEAF) methodology
- [ ] CEAF shows the maximum expected net benefit across all strategies at each WTP threshold, with uncertainty
- [ ] Contrast with CEAC (which shows probability each strategy is optimal) — CEAF is more informative for decision-making
- [ ] Design API: `voiage.plot.ceaf.plot_ceaf()` and/or `DecisionAnalysis.ceaf()`

### 3.2 Implement CEAF calculation [PENDING]
- [ ] Create `voiage/methods/ceaf.py` or add to existing module
- [ ] Calculate CEAF: for each WTP, find the strategy with maximum expected NB and its credible interval
- [ ] Compute CEAF uncertainty bands (e.g., 95% credible interval around the frontier)
- [ ] Write unit tests verifying:
  - CEAF ≥ CEAC values (frontier ≥ individual probabilities)
  - CEAF converges to 1.0 as uncertainty → 0
  - CEAF correctly identifies the optimal strategy at each WTP

### 3.3 Implement CEAF plotting [PENDING]
- [ ] Create `voiage/plot/ceaf.py`
- [ ] `plot_ceaf()` function with:
  - CEAF curve (frontier of max expected NB)
  - Uncertainty bands (shaded region)
  - Optimal strategy annotations
  - Customizable colors, labels, thresholds
- [ ] Write unit tests for plotting function
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90%

---

## Phase 4: Implement Extended Dominance Analysis [PENDING] [checkpoint: ]

### 4.1 Research and design [PENDING]
- [ ] Study extended dominance (strong and weak dominance) in cost-effectiveness analysis
- [ ] Strong dominance: strategy A dominates B if A is both more effective and less costly
- [ ] Extended dominance: strategy A extended-dominates B if the ICER of A vs C is lower than B vs A (for some C)
- [ ] Design API: `voiage.analysis.dominance()` and/or `DecisionAnalysis.dominance_analysis()`

### 4.2 Implement dominance analysis [PENDING]
- [ ] Create `voiage/analysis/dominance.py`
- [ ] `calculate_dominance()` — identify strongly dominated strategies
- [ ] `calculate_extended_dominance()` — identify extended-dominated strategies
- [ ] `calculate_icers()` — calculate incremental cost-effectiveness ratios
- [ ] `cost_effectiveness_frontier()` — identify strategies on the efficient frontier
- [ ] Return structured results: dominated strategies, frontier strategies, ICER table
- [ ] Write unit tests verifying against known examples from health economics literature
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90%

### 4.3 Implement dominance plotting [PENDING]
- [ ] Create `voiage/plot/dominance.py`
- [ ] `plot_cost_effectiveness_plane()` — scatter plot with dominance annotations
- [ ] `plot_icer_table()` — formatted ICER table visualization
- [ ] Write unit tests for plotting functions
- [ ] Ensure coverage ≥90%

---

## Phase 5: Implement Value of Heterogeneity [PENDING] [checkpoint: ]

### 5.1 Research and design [PENDING]
- [ ] Study Value of Heterogeneity (VOH) methodology — quantifies the value of tailoring decisions to subgroups
- [ ] VOH = Expected value of subgroup-specific decisions vs. one-size-fits-all decision
- [ ] Design API: `voiage.methods.heterogeneity.value_of_heterogeneity()`
- [ ] Reference: Basu & Meltzer (2007), Willan et al. methods

### 5.2 Implement Value of Heterogeneity [PENDING]
- [ ] Create `voiage/methods/heterogeneity.py`
- [ ] `value_of_heterogeneity()` — calculate VOH for subgroup-stratified PSA data
- [ ] `identify_optimal_subgroups()` — determine which subgroups benefit from personalized decisions
- [ ] `plot_heterogeneity_surface()` — visualize VOH across WTP and subgroup dimensions
- [ ] Support for multiple subgroup definitions (categorical and continuous)
- [ ] Write unit tests with synthetic subgroup data
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90%

### 5.3 Implement Value of Heterogeneity plotting [PENDING]
- [ ] Add `voiage/plot/heterogeneity.py` or extend existing plotting module
- [ ] `plot_voh_by_subgroup()` — bar chart of VOH per subgroup
- [ ] `plot_voh_heatmap()` — VOH across WTP × subgroup parameter space
- [ ] Write unit tests for plotting functions
- [ ] Ensure coverage ≥90%

---

## Phase 6: Portfolio Dynamic Programming [PENDING] [checkpoint: ]

### 6.1 Implement DP method for portfolio VOI [PENDING]
- [ ] **Current state:** `dynamic_programming()` in `voiage/methods/portfolio.py` raises `VoiageNotImplementedError`
- [ ] **Required:** Implement actual dynamic programming algorithm for optimal research portfolio selection
- [ ] Algorithm: backward induction over research studies, selecting optimal subset under budget constraint
- [ ] Support for:
  - Budget-constrained portfolio optimization
  - Inter-study dependencies (studies that reduce shared parameter uncertainty)
  - Sequential portfolio decisions (study results inform subsequent study choices)
- [ ] Write unit tests with synthetic portfolio data
- [ ] Ensure strict ruff/mypy compliance
- [ ] Ensure coverage ≥90%

---

## Phase 7: Full Verification [checkpoint: ]

### 7.1 Run complete tooling suite [PENDING]
- [ ] `ruff check voiage/ tests/` — zero errors
- [ ] `mypy voiage/ --strict` — zero errors
- [ ] `pytest tests/ --cov=voiage --cov-fail-under=90` — all tests pass, coverage ≥90%

### 7.2 Validate new methods against published results [PENDING]
- [ ] Efficient EVSI: compare against two-loop MC for accuracy, benchmark speedup
- [ ] CEAF: verify against BCEA R package results
- [ ] Extended dominance: verify against published ICER tables
- [ ] Value of Heterogeneity: verify against published VOH examples
- [ ] Document any discrepancies and explain sources

### 7.3 Integration tests for new methods [PENDING]
- [ ] Test all new methods through `DecisionAnalysis` class
- [ ] Test end-to-end workflows combining multiple methods
- [ ] Test edge cases: small samples, high correlation, degenerate inputs

---

## Phase 8: Autonomous Track Review, Archive and Progression [checkpoint: ]

### 8.1 Phase Review Protocol — Execute after EVERY phase (1-7) [PENDING]
After completing each phase above, execute the following protocol **before** marking the phase `[x]` and proceeding:

1. **Commit phase changes:**
   - `git add -A`
   - `git commit -m "conductor(track3): Complete phase <N> of implement-missing-methods"`
2. **Invoke `/conductor:review`** targeting all changes since the previous checkpoint commit.
3. **Apply all Critical and High severity fixes** identified by the review automatically.
4. **Re-run verification:** `ruff check voiage/ tests/ && mypy voiage/ --strict && pytest tests/ --cov=voiage --cov-fail-under=90 -q`
5. **If failures persist after 2 fix attempts:** Halt and report to user with details.
6. **Commit review fixes:** `git add -A && git commit -m "fix(conductor): Apply automated review fixes for phase <N>"`
7. **Mark phase complete** in this plan file (change `[PENDING]` → `[x]`).

### 8.2 Track Completion Protocol — Execute after final phase [PENDING]
After Phase 7 is complete and all phase reviews pass:

1. **Invoke `/conductor:review`** targeting the **entire track** (from track start commit to HEAD).
2. **Apply all Critical, High, and Medium severity fixes** automatically.
3. **Re-run full test suite:** `ruff check voiage/ tests/ && mypy voiage/ --strict && pytest tests/ --cov=voiage --cov-fail-under=90`
4. **Commit review fixes:** `git add -A && git commit -m "fix(conductor): Apply final track review fixes for implement-missing-methods"`
5. **Archive the track:**
   - `mkdir -p conductor/archive && mv conductor/tracks/implement-missing-methods conductor/archive/implement-missing-methods`
   - Update `conductor/tracks.md`: change `[ ]` → `[x]` for this track, add `[completed: <date>]` and archive link.
6. **Commit archive:** `git add -A conductor/ && git commit -m "chore(conductor): Archive completed track implement-missing-methods"`
7. **Read next track:** Read `conductor/tracks/cli-integration-testing/plan.md` and begin execution from Phase 1, Task 1.1.
8. **Announce:** "Track 3 (implement-missing-methods) complete. Starting Track 4 (cli-integration-testing)."
