# Track Implementation Plan: Activate and Wire Up Public API

## Phase 1: Wire Up `voiage/methods/__init__.py` [checkpoint: ]

### 1.1 Uncomment and verify all method imports [PENDING]
- [ ] Uncomment `from .structural import structural_evpi, structural_evppi` — verify functions exist
- [ ] Uncomment `from .network_nma import evsi_nma` — verify function exists
- [ ] Uncomment `from .adaptive import adaptive_evsi` — verify function exists
- [ ] Uncomment `from .portfolio import portfolio_voi` — verify function name is correct
- [ ] Uncomment `from .sequential import sequential_voi` — verify function name is correct
- [ ] Uncomment `from .observational import voi_observational` — verify function name is correct
- [ ] Uncomment `from .calibration import voi_calibration` — verify function name is correct
- [ ] Run `python -c "from voiage.methods import structural_evpi, structural_evppi, evsi_nma, adaptive_evsi, sequential_voi, voi_observational, voi_calibration"` — clean import
- [ ] Fix any broken import names (function names may differ from commented imports)

### 1.2 Wire up basic methods [PENDING]
- [ ] Verify `from voiage.methods.basic import evpi, evppi` works
- [ ] Verify `from voiage.methods.sample_information import evsi, enbs` works
- [ ] Create `__all__` list in `voiage/methods/__init__.py` for explicit public API

### 1.3 Ensure strict ruff/mypy compliance [PENDING]
- [ ] New import code must pass strict ruff (`ruff check voiage/methods/__init__.py`)
- [ ] New import code must pass strict mypy (`mypy voiage/methods/__init__.py`)
- [ ] Add module-level docstring following NumPy docstring convention

---

## Phase 2: Wire Up `voiage/plot/__init__.py` [checkpoint: ]

### 2.1 Uncomment and verify plotting imports [PENDING]
- [ ] Uncomment `from .voi_curves import ...` — verify function names match actual exports
- [ ] Uncomment `from .ceac import ...` — verify function names match actual exports
- [ ] Read actual function signatures in `voi_curves.py` and `ceac.py` to confirm export names
- [ ] Run `python -c "from voiage.plot import plot_ceac, plot_evpi_vs_wtp, plot_evsi_vs_sample_size, plot_evppi_surface"` — clean import
- [ ] Fix any broken import names

### 2.2 Create `__all__` and module docstring [PENDING]
- [ ] Create `__all__` list in `voiage/plot/__init__.py`
- [ ] Add module-level docstring
- [ ] Ensure matplotlib/seaborn import errors are handled gracefully with clear messages

### 2.3 Ensure strict ruff/mypy compliance [PENDING]
- [ ] New import code must pass strict ruff
- [ ] New import code must pass strict mypy

---

## Phase 3: Wire Up `voiage/core/__init__.py` [checkpoint: ]

### 3.1 Decide fate of `voiage/core/` [PENDING]
- [ ] Determine: should `voiage/core/` remain as utility subpackage or be removed?
- [ ] Current state: imports are commented out, `data_structures` module doesn't exist (moved to `voiage/schema.py`)
- [ ] **Recommended:** Keep `voiage/core/` for IO and validation utilities only:
  - `io.py` → CSV read/write for PSA samples and net benefit arrays
  - `utils.py` → input validation, net benefit calculation helpers
- [ ] Remove commented-out `data_structures` imports (schema.py is canonical source)

### 3.2 Uncomment and verify core utility imports [PENDING]
- [ ] Uncomment `from .io import ...` — verify functions exist and work
- [ ] Uncomment `from .utils import ...` — verify functions exist and work
- [ ] Create `__all__` list
- [ ] Add module-level docstring

### 3.3 Ensure strict ruff/mypy compliance [PENDING]
- [ ] New import code must pass strict ruff
- [ ] New import code must pass strict mypy

---

## Phase 4: Create Top-Level Public API [checkpoint: ]

### 4.1 Populate `voiage/__init__.py` [PENDING]
- [ ] Add module-level docstring describing the library
- [ ] Add `__version__` from package metadata
- [ ] Import and re-export key classes:
  ```python
  from voiage.analysis import DecisionAnalysis
  from voiage.schema import (
      ValueArray,
      ParameterSet,
      TrialDesign,
      DecisionOption,
      PortfolioStudy,
      PortfolioSpec,
  )
  from voiage.methods.basic import evpi, evppi
  from voiage.methods.sample_information import evsi, enbs
  ```
- [ ] Create `__all__` list for explicit public API surface

### 4.2 Verify top-level imports work [PENDING]
- [ ] `python -c "from voiage import evpi, evppi, evsi, enbs"` — clean
- [ ] `python -c "from voiage import DecisionAnalysis"` — clean
- [ ] `python -c "from voiage import ValueArray, ParameterSet, TrialDesign"` — clean
- [ ] `python -c "from voiage import DecisionAnalysis; da = DecisionAnalysis(...)"` — works

### 4.3 Ensure strict ruff/mypy compliance [PENDING]
- [ ] `voiage/__init__.py` must pass strict ruff
- [ ] `voiage/__init__.py` must pass strict mypy

---

## Phase 5: Full Verification [checkpoint: ]

### 5.1 Run complete tooling suite [PENDING]
- [ ] `ruff check voiage/ tests/` — zero errors
- [ ] `mypy voiage/ --strict` — zero errors
- [ ] `pytest tests/ --cov=voiage --cov-fail-under=90` — all tests pass, coverage ≥90%

### 5.2 Verify no regressions [PENDING]
- [ ] All previously-passing tests still pass
- [ ] No new test failures from import changes
- [ ] CLI commands still work: `voiage --help`, `voiage calculate-evpi --help`, etc.

### 5.3 Coverage check for newly-wired modules [PENDING]
- [ ] Verify `voiage/methods/__init__.py` has import tests
- [ ] Verify `voiage/plot/__init__.py` has import tests
- [ ] Verify `voiage/__init__.py` has import tests
- [ ] Overall project coverage must still be ≥90%
