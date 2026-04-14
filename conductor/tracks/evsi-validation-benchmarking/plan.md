# Track Implementation Plan: EVSI/EVPPI Validation & Benchmarking

## Phase 1: EVSI Validation Against Published Results [checkpoint: ]

### 1.1 Research and gather published EVSI results [PENDING]
- [ ] Identify 2-3 published health economics studies with reported EVSI values
- [ ] Document the study parameters, model specifications, and reported EVSI results
- [ ] Sources: ISPOR guidelines, NICE DSU technical support documents, published papers

### 1.2 Create EVSI validation notebook [PENDING]
- [ ] Create `examples/evsi_validation.ipynb`
- [ ] Implement the published study models using voiage's API
- [ ] Compare voiage's EVSI results against published values
- [ ] Document any discrepancies and explain sources of variation
- [ ] Target: Results within 5% of published values (accounting for Monte Carlo variance)

### 1.3 Create EVPPI validation notebook [PENDING]
- [ ] Create `examples/evppi_validation.ipynb`
- [ ] Validate regression-based EVPPI against published results
- [ ] Compare multiple regression methods (GAM, RandomForest, BART if available)
- [ ] Document accuracy and computational efficiency of each method

---

## Phase 2: Integration Testing [checkpoint: ]

### 2.1 DecisionAnalysis integration tests for EVSI [PENDING]
- [ ] Create `tests/test_evsi_integration.py`
- [ ] Test EVSI through `DecisionAnalysis.evsi()` method (two-loop)
- [ ] Test EVSI through `DecisionAnalysis.evsi()` method (regression)
- [ ] Test ENBS calculation through DecisionAnalysis
- [ ] Test EVSI with population scaling parameters

### 2.2 DecisionAnalysis integration tests for EVPPI [PENDING]
- [ ] Add EVPPI integration tests to existing test file or create new
- [ ] Test EVPPI through `DecisionAnalysis.evppi()` with multiple regression methods
- [ ] Test EVPPI with multiple parameter subsets
- [ ] Test EVPPI edge cases (single parameter, all parameters, correlated parameters)

### 2.3 NMA EVSI integration tests [PENDING]
- [ ] Create `tests/test_nma_evsi_integration.py`
- [ ] Test `evsi_nma()` through DecisionAnalysis
- [ ] Test with realistic NMA network data (3+ treatments)
- [ ] Verify Bayesian updating produces sensible posterior estimates

### 2.4 Cross-method consistency tests [PENDING]
- [ ] Test that EVSI → 0 as sample size → 0
- [ ] Test that EVSI → EVPI as sample size → ∞ (asymptotic behavior)
- [ ] Test that EVPPI ≤ EVPI for any parameter subset
- [ ] Test that ENBS = EVSI - research_cost (identity check)

---

## Phase 3: Performance Benchmarking [checkpoint: ]

### 3.1 NumPy vs JAX EVSI benchmark [PENDING]
- [ ] Create benchmark script: `benchmarks/benchmark_evsi_numpy_jax.py`
- [ ] Benchmark two-loop EVSI across varying sample sizes (100, 1000, 10000)
- [ ] Benchmark regression EVSI across varying outer loop sizes
- [ ] Record wall-clock time, memory usage, and result accuracy
- [ ] Save results to `benchmarks/results/evsi_performance.json`

### 3.2 JAX JIT compilation benchmark [PENDING]
- [ ] Benchmark JIT-compiled vs non-JIT EVSI
- [ ] Measure compilation time vs execution time tradeoff
- [ ] Identify break-even point where JIT becomes worthwhile

### 3.3 Regression method comparison [PENDING]
- [ ] Benchmark GAM vs RandomForest vs BART for regression-based EVSI
- [ ] Compare accuracy, speed, and robustness across methods
- [ ] Document which method is best for which use case

### 3.4 Parallel processing benchmark [PENDING]
- [ ] Benchmark `voiage.parallel.monte_carlo` with varying process counts
- [ ] Measure speedup vs overhead for parallel EVSI computation
- [ ] Identify optimal process count for different problem sizes

---

## Phase 4: Documentation and Reporting [checkpoint: ]

### 4.1 Performance report [PENDING]
- [ ] Create `docs/user_guide/performance/evsi_performance.md`
- [ ] Document NumPy vs JAX performance characteristics
- [ ] Include benchmark results tables and charts
- [ ] Provide recommendations for method selection

### 4.2 Validation report [PENDING]
- [ ] Create `docs/user_guide/validation/evsi_validation.md`
- [ ] Summarize validation results against published studies
- [ ] Document accuracy characteristics of each method
- [ ] Provide guidance on choosing Monte Carlo parameters

### 4.3 Method selection guide [PENDING]
- [ ] Create `docs/user_guide/guides/method_selection.md`
- [ ] Decision tree for choosing EVSI method (two-loop vs regression)
- [ ] Decision tree for choosing EVPPI regression method
- [ ] Performance vs accuracy tradeoff guidance
