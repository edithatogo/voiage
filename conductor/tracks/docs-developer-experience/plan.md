# Track Implementation Plan: Documentation, Validation and Developer Experience

## Phase 1: Complete Public API Docstrings [checkpoint: ]

### 1.1 Audit existing docstrings [PENDING]
- [ ] Review all public functions in `voiage/methods/` — list those missing or incomplete docstrings
- [ ] Review all public functions in `voiage/plot/` — list those missing or incomplete docstrings
- [ ] Review all classes in `voiage/schema.py`, `voiage/analysis.py` — list those missing or incomplete docstrings
- [ ] Review all modules — list those missing module-level docstrings

### 1.2 Standardize docstring format [PENDING]
- [ ] All docstrings follow NumPy convention (per `[tool.ruff.lint.pydocstyle].convention = "numpy"`)
- [ ] Every public function docstring includes:
  - One-line summary
  - Extended description (for complex methods)
  - **Parameters** section with name, type, description
  - **Returns** section with type and description
  - **Raises** section listing all exceptions
  - **Examples** section with runnable code
  - **Notes** section for mathematical formulas (VOI methods)
  - **References** section citing the original papers
- [ ] Every public class docstring includes:
  - Class purpose and usage summary
  - **Attributes** section
  - **Examples** section

### 1.3 Add mathematical documentation [PENDING]
- [ ] EVPI: document formula E[max_i NB_i(θ)] - max_i E[NB_i(θ)]
- [ ] EVPPI: document regression-based approach and metamodel types
- [ ] EVSI (two-loop): document nested Monte Carlo formula
- [ ] EVSI (efficient): document Heath et al. method
- [ ] EVSI (moment-based): document variance reduction formula
- [ ] ENBS: document EVSI - research_cost
- [ ] Structural VOI: document model uncertainty framework
- [ ] NMA VOI: document network meta-analysis VOI formulas
- [ ] CEAF: document frontier methodology
- [ ] Extended dominance: document ICER calculations
- [ ] Value of Heterogeneity: document subgroup analysis formulas
- [ ] Use reStructuredMath or LaTeX notation within docstrings

### 1.4 Verify docstring rendering [PENDING]
- [ ] Build Sphinx docs: `sphinx-build -b html docs docs/_build/html`
- [ ] Verify all docstrings render correctly in API reference
- [ ] Fix any Sphinx warnings or rendering issues

---

## Phase 2: Validation Notebooks [checkpoint: ]

### 2.1 EVPI validation notebook [PENDING]
- [ ] Create `examples/evpi_validation.ipynb`
- [ ] Validate against published EVPI results from:
  - BCEA R package vignette examples
  - NICE DSU Technical Support Document
  - Published health economics case studies
- [ ] Compare voiage EVPI against published values — target: within 1% (Monte Carlo variance accounted for)
- [ ] Document any discrepancies and explain sources

### 2.2 EVPPI validation notebook [PENDING]
- [ ] Create `examples/evppi_validation.ipynb`
- [ ] Validate regression-based EVPPI against:
  - Strong Oakley method (gold standard)
  - Published EVPPI results from literature
- [ ] Compare multiple regression methods (linear, GAM, RF, BART) for accuracy
- [ ] Document accuracy characteristics of each method
- [ ] Target: regression EVPPI within 5% of Monte Carlo EVPPI

### 2.3 EVSI validation notebook [PENDING]
- [ ] Create `examples/evsi_validation.ipynb`
- [ ] Validate two-loop EVSI against published results
- [ ] Validate efficient/moment-based EVSI against two-loop MC
- [ ] Compare accuracy and speed across all methods
- [ ] Target: efficient EVSI within 5% of two-loop, 10-100x faster

### 2.4 NMA VOI validation notebook [PENDING]
- [ ] Update existing `examples/nma_validation.ipynb` or create new
- [ ] Validate NMA EVPI/EVPPI against published network meta-analysis examples
- [ ] Use a realistic multi-treatment network (≥4 treatments)
- [ ] Document consistency checking results

### 2.5 Structural VOI validation notebook [PENDING]
- [ ] Create `examples/structural_voi_validation.ipynb`
- [ ] Validate structural EVPI/EVPPI against model comparison studies
- [ ] Demonstrate with a realistic model uncertainty scenario
- [ ] Document the structural uncertainty framework

---

## Phase 3: Tutorial Notebooks [checkpoint: ]

### 3.1 Getting Started tutorial [PENDING]
- [ ] Create `examples/getting_started.ipynb`
- [ ] Walk through:
  1. Installation and setup
  2. Loading PSA data
  3. Basic EVPI calculation
  4. EVPPI for key parameters
  5. EVSI for proposed study
  6. Plotting results (CEAC, VOI curves)
  7. Interpreting results for decision-making
- [ ] Use a realistic health economics example (e.g., cancer treatment comparison)
- [ ] Target: new user can complete tutorial in 30 minutes
- [ ] Include exercises with solutions

### 3.2 Advanced methods tutorial [PENDING]
- [ ] Create `examples/advanced_methods.ipynb`
- [ ] Cover:
  1. Structural VOI — when model structure is uncertain
  2. NMA VOI — comparing multiple treatments in a network
  3. Adaptive trial VOI — value of interim analyses
  4. Portfolio VOI — selecting optimal research portfolio
  5. Sequential VOI — value of waiting for more information
  6. Efficient EVSI — fast approximation methods
  7. CEAF and extended dominance — advanced decision analysis
- [ ] Include realistic examples for each method
- [ ] Compare methods: when to use which

### 3.3 Domain-specific tutorials [PENDING]
- [ ] Create `examples/financial_voi.ipynb`:
  - Portfolio optimization for investment decisions
  - Value of market research
  - Value of reducing parameter uncertainty in financial models
- [ ] Create `examples/environmental_voi.ipynb`:
  - Value of environmental monitoring data
  - Value of reducing uncertainty in climate models
  - Policy decision analysis
- [ ] Create `examples/engineering_voi.ipynb` (if applicable):
  - Value of additional testing in engineering design
  - Value of reducing uncertainty in reliability models

### 3.4 JAX performance tutorial [PENDING]
- [ ] Create `examples/jax_performance.ipynb`
- [ ] Demonstrate:
  1. NumPy vs JAX performance comparison
  2. JIT compilation benefits
  3. GPU acceleration (if available)
  4. When JAX is worth the setup cost
- [ ] Include benchmark results and recommendations

---

## Phase 4: User Guide Documentation [checkpoint: ]

### 4.1 Update existing documentation [PENDING]
- [ ] Review and update `docs/getting_started.rst` — ensure accuracy
- [ ] Review and update `docs/installation.rst` — ensure accuracy
- [ ] Review and update `docs/introduction.rst` — ensure accuracy
- [ ] Update all outdated code examples
- [ ] Ensure all documented methods are actually importable

### 4.2 Create missing documentation pages [PENDING]
- [ ] Create `docs/methods/evpi.rst`
- [ ] Create `docs/methods/evppi.rst`
- [ ] Create `docs/methods/evsi.rst` (all methods: two_loop, regression, efficient, moment_based)
- [ ] Create `docs/methods/enbs.rst`
- [ ] Create `docs/methods/structural_voi.rst`
- [ ] Create `docs/methods/network_nma.rst`
- [ ] Create `docs/methods/adaptive_trials.rst`
- [ ] Create `docs/methods/portfolio_optimization.rst`
- [ ] Create `docs/methods/sequential_voi.rst`
- [ ] Create `docs/methods/observational_voi.rst`
- [ ] Create `docs/methods/calibration_voi.rst`
- [ ] Create `docs/methods/ceaf.rst`
- [ ] Create `docs/methods/dominance.rst`
- [ ] Create `docs/methods/heterogeneity.rst`
- [ ] Create `docs/plotting/index.rst`
- [ ] Create `docs/cli_reference.rst`
- [ ] Create `docs/data_structures.rst`
- [ ] Create `docs/backends.rst` (NumPy vs JAX)

### 4.3 API reference generation [PENDING]
- [ ] Configure Sphinx autodoc to capture all public API
- [ ] Ensure `docs/conf.py` has proper `autodoc_mock_imports` for third-party libs
- [ ] Verify all modules appear in generated API docs
- [ ] Test that `make html` builds without warnings
- [ ] Verify API docs render correctly on Read the Docs / GitHub Pages

### 4.4 CLI documentation [PENDING]
- [ ] Create `docs/cli_reference.rst` with all commands
- [ ] Document every CLI subcommand with examples and expected output
- [ ] Create sample input data files for CLI examples
- [ ] Document common error messages and solutions

---

## Phase 5: Developer Onboarding [checkpoint: ]

### 5.1 Update CONTRIBUTING.md [PENDING]
- [ ] Ensure setup instructions are current and accurate
- [ ] Document the test running commands
- [ ] Document the ruff strict configuration
- [ ] Document the mypy strict configuration
- [ ] Document the coverage requirements (≥90%)
- [ ] Add section on how to add new VOI methods
- [ ] Add section on how to add new plotting functions

### 5.2 Create architecture documentation [PENDING]
- [ ] Create `docs/developer_guide/architecture.rst`
- [ ] Document the module structure and dependencies (with diagram)
- [ ] Document the backend abstraction (NumPy/JAX)
- [ ] Document the factory pattern for method instantiation
- [ ] Document the DecisionAnalysis class design
- [ ] Document the data flow: input → schema → method → output → plot

### 5.3 Create "How to Contribute" guide [PENDING]
- [ ] Create `docs/developer_guide/how_to_contribute.rst`
- [ ] Step-by-step guide for first-time contributors
- [ ] Explain the conductor workflow for AI-assisted development
- [ ] Explain the branching and PR process
- [ ] Explain the testing and coverage requirements

### 5.4 Create method implementation guide [PENDING]
- [ ] Create `docs/developer_guide/implementing_new_methods.rst`
- [ ] Step-by-step guide for adding a new VOI method
- [ ] Template for method function signature
- [ ] Checklist: implementation, tests, docs, CLI, validation

---

## Phase 6: Final Polish [checkpoint: ]

### 6.1 README.md overhaul [PENDING]
- [ ] Update with current feature set (mark everything actually implemented)
- [ ] Add quick-start example that works out of the box
- [ ] Add working badges: PyPI version, CI status, coverage, Python versions
- [ ] Add comparison table vs R packages (BCEA, dampack, voi)
- [ ] Add links to all key documentation pages
- [ ] Add "Why voiage?" section explaining the project's unique value
- [ ] Add "Getting Help" section (discussions, issues, contact)

### 6.2 Changelog update [PENDING]
- [ ] Update `changelog.md` with all changes from Tracks 1-5
- [ ] Organize by category: Added, Changed, Fixed, Removed
- [ ] Include breaking changes section
- [ ] Include migration guide for breaking changes

### 6.3 Todo.md and Roadmap finalization [PENDING]
- [ ] Move all completed items to "Done" section
- [ ] Remove items that are obsolete
- [ ] Add any remaining follow-up tasks
- [ ] Update roadmap.md to reflect completed state
- [ ] Propose next major release targets (v0.3, v0.4, v1.0)

### 6.4 Project presentation [PENDING]
- [ ] Review all public-facing documentation for consistency
- [ ] Ensure all examples in docs actually run and produce stated output
- [ ] Run all tutorial notebooks end-to-end to verify they work
- [ ] Verify GitHub Pages site builds and renders correctly
- [ ] Check all external links in documentation

---

## Phase 7: Full Verification [checkpoint: ]

### 7.1 Run complete tooling suite [PENDING]
- [ ] `ruff check voiage/ tests/` — zero errors
- [ ] `mypy voiage/ --strict` — zero errors
- [ ] `pytest tests/ --cov=voiage --cov-fail-under=90` — all tests pass, coverage ≥90%

### 7.2 Verify documentation builds [PENDING]
- [ ] `sphinx-build -b html docs docs/_build/html` — zero warnings
- [ ] Verify all links in documentation work
- [ ] Verify all code examples in documentation run correctly

### 7.3 Final end-to-end verification [PENDING]
- [ ] Fresh install from `pip install voiage` (or local equivalent)
- [ ] Run getting started tutorial — works without modification
- [ ] Run all tutorial notebooks — all execute without errors
- [ ] Run all CLI commands — all work as documented
- [ ] Report any issues found
