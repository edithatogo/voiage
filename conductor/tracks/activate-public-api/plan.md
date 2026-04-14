# Track Implementation Plan: Activate and Wire Up Public API

## Phase 1: Audit and Inventory [checkpoint: ]

### 1.1 Audit all commented-out imports [COMPLETED]
- [x] **Audit `voiage/methods/__init__.py`** — Identify all commented import lines
- [x] **Audit `voiage/plot/__init__.py`** — Identify all commented import lines
- [x] **Audit `voiage/core/__init__.py`** — Identify all commented import lines
- [x] **Audit `voiage/__init__.py`** — Check current state (likely empty)

### 1.2 Verify underlying modules exist and are functional [COMPLETED]
- [x] **Verify `voiage/methods/structural.py`** exists with `structural_evpi`, `structural_evppi`
- [x] **Verify `voiage/methods/network_nma.py`** exists with `evsi_nma`
- [x] **Verify `voiage/methods/adaptive.py`** exists with `adaptive_evsi`
- [x] **Verify `voiage/methods/portfolio.py`** exists
- [x] **Verify `voiage/methods/sequential.py`** exists
- [x] **Verify `voiage/methods/observational.py`** exists
- [x] **Verify `voiage/methods/calibration.py`** exists
- [x] **Verify `voiage/plot/ceac.py`** exists with `plot_ceac`
- [x] **Verify `voiage/plot/voi_curves.py`** exists with plotting functions
- [x] **Verify `voiage/schema.py`** is the canonical source for data structures

---

## Phase 2: Wire Up Methods Package [checkpoint: ]

### 2.1 Activate `voiage/methods/__init__.py` [PENDING]
- [ ] Uncomment imports for `structural_evpi`, `structural_evppi`
- [ ] Uncomment import for `evsi_nma`
- [ ] Uncomment import for `adaptive_evsi`
- [ ] Uncomment import for `portfolio_voi` (or correct function names)
- [ ] Uncomment import for `sequential_voi` (or correct function names)
- [ ] Uncomment import for `voi_observational` (or correct function names)
- [ ] Uncomment import for `voi_calibration` (or correct function names)
- [ ] Verify each import resolves to an actual function (fix names if needed)
- [ ] Run `python -c "from voiage.methods import *"` to verify clean import

### 2.2 Wire up `voiage/methods/sample_information.py` [PENDING]
- [ ] Verify `evsi()` function is properly exported
- [ ] Verify `enbs()` function is properly exported
- [ ] Ensure regression method raises proper `VoiageNotImplementedError` when sklearn unavailable

### 2.3 Wire up `voiage/methods/basic.py` [PENDING]
- [ ] Verify `evpi()` function is properly exported
- [ ] Verify `evppi()` function is properly exported

---

## Phase 3: Wire Up Plot Package [checkpoint: ]

### 3.1 Activate `voiage/plot/__init__.py` [PENDING]
- [ ] Uncomment imports from `voi_curves` module
- [ ] Uncomment imports from `ceac` module
- [ ] Verify `plot_ceac` resolves correctly
- [ ] Verify `plot_evpi_vs_wtp`, `plot_evsi_vs_sample_size`, `plot_evppi_surface` resolve correctly
- [ ] Run `python -c "from voiage.plot import *"` to verify clean import

### 3.3 Test plotting functions end-to-end [PENDING]
- [ ] Verify plotting tests pass: `pytest tests/test_plotting.py -v`
- [ ] Fix any plotting issues (matplotlib/seaborn compatibility)

---

## Phase 4: Wire Up Core Package [checkpoint: ]

### 4.1 Clean up `voiage/core/__init__.py` [PENDING]
- [ ] Determine if `voiage/core/` should remain as a subpackage or be removed
- [ ] If keeping: Uncomment and fix imports for IO utilities and validation functions
- [ ] If removing: Update any references to `voiage.core` elsewhere in codebase
- [ ] **Decision:** Since `voiage/schema.py` is the canonical source for data structures, `voiage/core/` should contain only utility functions (IO, validation) — update accordingly

### 4.2 Verify core utilities work [PENDING]
- [ ] Test `voiage/core/io.py` functions (CSV read/write for PSA samples and net benefit arrays)
- [ ] Test `voiage/core/utils.py` functions (input validation, net benefit calculation)

---

## Phase 5: Create Clean Top-Level API [checkpoint: ]

### 5.1 Populate `voiage/__init__.py` [PENDING]
- [ ] Import and re-export key functions for clean top-level access:
  ```python
  from voiage.analysis import DecisionAnalysis
  from voiage.schema import ValueArray, ParameterSet, TrialDesign, DecisionOption
  from voiage.methods.basic import evpi, evppi
  from voiage.methods.sample_information import evsi, enbs
  ```
- [ ] Include `__version__` from package metadata
- [ ] Include `__all__` list for explicit public API

### 5.2 Create convenience module re-exports [PENDING]
- [ ] Ensure `voiage.methods` exposes all method functions at submodule level
- [ ] Ensure `voiage.plot` exposes all plotting functions at submodule level
- [ ] Ensure `voiage.schema` exposes all data structures at submodule level

---

## Phase 6: Full Verification and Cleanup [checkpoint: ]

### 6.1 Run complete test suite [PENDING]
- [ ] Run `tox` to verify linting, type checking, and tests all pass
- [ ] Fix any failures from newly wired imports
- [ ] Address any mypy type errors from newly active code paths

### 6.2 Fix commented-out tests [PENDING]
- [ ] Check for any test files with commented-out test code
- [ ] Uncomment and fix tests to work with wired-up imports
- [ ] Ensure all tests in `tests/test_sample_information.py` pass

### 6.3 Update pyproject.toml exclusions [PENDING]
- [ ] Review `--ignore` patterns in pytest config — remove ignores for now-active test files
- [ ] Review ruff/mypy exclusions — remove exclusions for now-active modules
- [ ] Review coverage exclusions — ensure all active modules are covered

### 6.4 Verify CLI still works [PENDING]
- [ ] Test `voiage --help` works
- [ ] Test `voiage evpi --help` and other CLI commands work with wired-up imports
- [ ] Verify CLI commands use the newly activated methods where applicable
