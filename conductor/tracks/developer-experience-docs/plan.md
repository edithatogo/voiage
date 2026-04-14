# Track Implementation Plan: Developer Experience & Documentation

## Phase 1: Complete Public API Docstrings [checkpoint: ]

### 1.1 Audit existing docstrings [PENDING]
- [ ] Review all public functions in `voiage/methods/` for docstring completeness
- [ ] Review all public functions in `voiage/plot/` for docstring completeness
- [ ] Review all classes in `voiage/schema.py` for docstring completeness
- [ ] Review `voiage/analysis.py` (DecisionAnalysis) for docstring completeness
- [ ] Create list of functions/classes missing or incomplete docstrings

### 1.2 Standardize docstring format [PENDING]
- [ ] Ensure all docstrings follow NumPy docstring convention (per CONTRIBUTING.md)
- [ ] Each docstring must include: summary, parameters (with types), returns, raises, examples
- [ ] Add `Examples` section to every public function with runnable code
- [ ] Verify docstrings render correctly in Sphinx documentation

### 1.3 Fix top-level API docstrings [PENDING]
- [ ] Add module-level docstring to `voiage/__init__.py`
- [ ] Add module-level docstrings to `voiage/methods/__init__.py`
- [ ] Add module-level docstrings to `voiage/plot/__init__.py`
- [ ] Add module-level docstrings to `voiage/schema.py`

---

## Phase 2: Tutorial Notebooks [checkpoint: ]

### 2.1 Getting Started tutorial [PENDING]
- [ ] Create `examples/getting_started.ipynb`
- [ ] Walk through: installation, basic EVPI calculation, EVPPI, EVSI
- [ ] Use a realistic health economics example (e.g., cancer treatment comparison)
- [ ] Include visualization with plotting functions
- [ ] Target: New user can complete tutorial in 30 minutes

### 2.2 Advanced EVSI tutorial [PENDING]
- [ ] Create `examples/advanced_evsi.ipynb`
- [ ] Compare two-loop vs regression methods
- [ ] Demonstrate population EVSI with scaling
- [ ] Show ENBS optimization for sample size determination
- [ ] Include JAX backend performance comparison

### 2.3 Network Meta-Analysis tutorial [PENDING]
- [ ] Update `examples/nma_validation.ipynb` or create `examples/network_meta_analysis.ipynb`
- [ ] Demonstrate NMA EVPI, EVPPI, and EVSI workflows
- [ ] Use realistic multi-treatment network data
- [ ] Show consistency checking and network visualization

### 2.4 Domain-specific tutorials [PENDING]
- [ ] Create `examples/financial_voi.ipynb` — VOI for investment decisions
- [ ] Create `examples/environmental_voi.ipynb` — VOI for environmental policy
- [ ] Demonstrate `voiage.multi_domain` capabilities

---

## Phase 3: CLI Documentation and Examples [checkpoint: ]

### 3.1 CLI reference documentation [PENDING]
- [ ] Update `docs/user_guide/cli_reference.md` with all commands
- [ ] Document every CLI subcommand with examples
- [ ] Include `voiage evpi`, `voiage evppi`, `voiage evsi`, `voiage enbs`
- [ ] Include `voiage calculate-structural-evpi`, `voiage calculate-nma-voi`
- [ ] Document all CLI flags and options

### 3.2 CLI example scripts [PENDING]
- [ ] Create `examples/cli_examples.sh` with copy-paste-ready commands
- [ ] Create sample input data files for CLI examples:
  - `examples/data/sample_psa_params.csv`
  - `examples/data/sample_net_benefits.csv`
- [ ] Document expected output for each example

### 3.3 CLI error handling documentation [PENDING]
- [ ] Document common CLI error messages and their solutions
- [ ] Create troubleshooting section for CLI usage

---

## Phase 4: User Guide Documentation [checkpoint: ]

### 4.1 Update existing user guide [PENDING]
- [ ] Review and update `docs/user_guide/` for accuracy
- [ ] Ensure all documented methods are actually importable
- [ ] Update any outdated code examples
- [ ] Add missing sections for newly activated methods

### 4.2 Create missing documentation pages [PENDING]
- [ ] Create `docs/user_guide/methods/structural_voi.md`
- [ ] Create `docs/user_guide/methods/network_nma.md`
- [ ] Create `docs/user_guide/methods/adaptive_trials.md`
- [ ] Create `docs/user_guide/methods/portfolio_optimization.md`
- [ ] Create `docs/user_guide/plotting/index.md`
- [ ] Create `docs/user_guide/data_structures.md`

### 4.3 API reference generation [PENDING]
- [ ] Ensure Sphinx autodoc captures all public API
- [ ] Configure `docs/conf.py` for proper API reference generation
- [ ] Verify all modules appear in generated API docs
- [ ] Test that `make html` builds without warnings

---

## Phase 5: Developer Onboarding [checkpoint: ]

### 5.1 Update CONTRIBUTING.md [PENDING]
- [ ] Ensure setup instructions are current and accurate
- [ ] Document the test running commands
- [ ] Document the tox workflow
- [ ] Add section on how to add new VOI methods

### 5.2 Create architecture documentation [PENDING]
- [ ] Create `docs/developer_guide/architecture.md`
- [ ] Document the module structure and dependencies
- [ ] Document the backend abstraction (NumPy/JAX)
- [ ] Document the factory pattern for method instantiation
- [ ] Document the DecisionAnalysis class design

### 5.3 Create "How to Contribute" guide [PENDING]
- [ ] Create `docs/developer_guide/how_to_contribute.md`
- [ ] Step-by-step guide for first-time contributors
- [ ] Explain the conductor workflow for AI-assisted development
- [ ] Explain the branching and PR process

---

## Phase 6: Final Polish [checkpoint: ]

### 6.1 README.md update [PENDING]
- [ ] Update project README with current feature set
- [ ] Add quick-start example that works out of the box
- [ ] Add badges (PyPI version, CI status, coverage)
- [ ] Link to all key documentation pages

### 6.2 Changelog update [PENDING]
- [ ] Update `changelog.md` with all changes from this track
- [ ] Organize by category (Methods, Plotting, CLI, Docs, etc.)

### 6.3 Todo.md cleanup [PENDING]
- [ ] Move all completed todo.md items to "Done" section
- [ ] Remove items that are now obsolete
- [ ] Add any remaining follow-up tasks identified during this track
