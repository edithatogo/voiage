# voiage To-Do List

## v0.1 Implementation

- [ ] **1. Project Setup**
    - [x] Install core dependencies from `pyproject.toml`.
    - [x] Configure and enable pre-commit hooks for `black`, `flake8`, and `mypy`.
- [ ] **2. Core Data Structures**
    - [x] Implement `NetBenefitArray` in `pyvoi/core/data_structures.py`.
    - [x] Implement `PSASample` in `pyvoi/core/data_structures.py`.
    - [x] Implement `TrialArm`, `TrialDesign`, `PortfolioStudy`, `PortfolioSpec`, and `DynamicSpec` in `pyvoi/core/data_structures.py`.
- [ ] **3. Core VOI Methods**
    - [x] Create `pyvoi/methods/basic.py`.
    - [x/o] Implement `evpi()` in `pyvoi/methods/basic.py`.
    - [x/o] Implement `evppi()` in `pyvoi/methods/basic.py`.
    - [x/o] Implement `evsi()` in `pyvoi/methods/sample_information.py`.
    - [x/o] Implement `enbs()` in `pyvoi/methods/sample_information.py`.
- [x] **4. Unit Tests**
    - [x] Write unit tests for all core data structures.
    - [x] Write unit tests for `evpi()`.
    - [x] Write unit tests for `evppi()`.
    - [x] Write unit tests for `evsi()`.
    - [x] Write unit tests for `enbs()`.
- [ ] **5. Documentation**
    - [ ] Write docstrings for all new functions and classes.
    - [ ] Create initial documentation for the `docs/` directory.
