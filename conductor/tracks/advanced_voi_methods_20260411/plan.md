# Implementation Plan: Advanced VOI Methods

## Phase 1: Structural Uncertainty VOI

### Task 1.1: Define Data Structures for Structural Uncertainty
- [x] Task: Create data structures for multiple model structures (already implemented in voiage/methods/structural.py) <sha>
    - [x] Define `ModelStructureEvaluator` type alias
    - [x] Use existing `ParameterSet` and `ValueArray` from schema.py
    - [x] Add validation for model structure compatibility
    - [x] Write unit tests for data structures

### Task 1.2: Implement Structural Uncertainty VOI Algorithm
- [x] Task: Implement core structural uncertainty VOI calculation (already implemented) <sha>
    - [x] `structural_evpi()` function implemented
    - [x] `structural_evppi()` function implemented
    - [x] Support model averaging approaches
    - [x] Population scaling with annuity factor
    - [x] Comprehensive input validation
    - [x] Unit tests in tests/test_structural.py

### Task 1.3: Add CLI Interface for Structural Uncertainty
- [x] Task: Create CLI commands for structural uncertainty VOI <e6e6ea0>
    - [x] Write failing tests for CLI commands (tests/test_structural_cli_e2e.py)
    - [x] Add `voiage calculate-structural-evpi` command
    - [x] Add `voiage calculate-structural-evppi` command
    - [x] Support JSON config input for model structures
    - [x] Add comprehensive help/examples
    - [x] Test CLI end-to-end (tests written, will run in CI)

### Task 1.4: Enhance Structural VOI with JAX Backend
- [x] Task: Add JAX-accelerated structural VOI <7761e9d>
    - [x] Write failing tests for JAX structural EVPI
    - [x] Implement `structural_evpi_jit()` in structural.py
    - [x] Implement `structural_evppi_jit()` in structural.py
    - [x] Benchmark JAX vs NumPy performance (tests verify correctness)
    - [x] Verify >80% test coverage

- [x] Task: Conductor - Automated Review 'Phase 1: Structural Uncertainty VOI' (Protocol in workflow.md) [checkpoint: 07abda3]
    - [x] Run `/conductor:review` for Phase 1 changes
    - [x] Apply Critical/High severity fixes automatically (none needed)
    - [x] Re-run tests after fixes
    - [x] Commit review fixes
    - [x] Push to remote and verify CI passes
    - [x] Create checkpoint commit with git note

## Phase 2: Network Meta-Analysis VOI

### Task 2.1: Define Data Structures for Network Meta-Analysis
- [ ] Task: Create data structures for NMA input
    - [ ] Write failing tests for NMA data structures
    - [ ] Define `NetworkMetaAnalysisData` class
    - [ ] Support multi-treatment comparisons
    - [ ] Add validation for network consistency
    - [ ] Write unit tests for NMA data structures

### Task 2.2: Implement NMA VOI Algorithm
- [ ] Task: Implement NMA-based VOI calculation
    - [ ] Write failing tests for NMA VOI method
    - [ ] Implement `calculate_nma_evpi()` function
    - [ ] Support EVPI/EVPPI for network comparisons
    - [ ] Add JAX-accelerated backend
    - [ ] Verify >80% test coverage

### Task 2.3: Add CLI Interface for NMA VOI
- [ ] Task: Create CLI command for NMA VOI
    - [ ] Write failing tests for CLI command
    - [ ] Add `voiage calculate-nma-voi` command
    - [ ] Support NMA result CSV input
    - [ ] Add comprehensive help/examples
    - [ ] Test CLI end-to-end

### Task 2.4: Write Documentation for NMA VOI
- [ ] Task: Document NMA VOI method
    - [ ] Add docstrings with examples
    - [ ] Create Jupyter notebook example
    - [ ] Update README feature table

- [ ] Task: Conductor - Automated Review 'Phase 2: Network Meta-Analysis VOI' (Protocol in workflow.md)
    - [ ] Run `/conductor:review` for Phase 2 changes
    - [ ] Apply Critical/High severity fixes automatically
    - [ ] Re-run tests after fixes
    - [ ] Commit review fixes
    - [ ] Push to remote and verify CI passes
    - [ ] Create checkpoint commit with git note

## Phase 3: Integration & Validation

### Task 3.1: Integration Testing
- [ ] Task: Test both methods with real-world datasets
    - [ ] Test structural uncertainty with health economics models
    - [ ] Test NMA VOI with clinical trial networks
    - [ ] Verify results against known benchmarks
    - [ ] Add integration tests

### Task 3.2: Performance Optimization
- [ ] Task: Optimize performance for large datasets
    - [ ] Benchmark current performance
    - [ ] Optimize bottlenecks
    - [ ] Verify JAX acceleration working
    - [ ] Add performance tests

### Task 3.3: Final Documentation & Cleanup
- [ ] Task: Finalize all documentation
    - [ ] Review and update all docstrings
    - [ ] Ensure examples work correctly
    - [ ] Update changelog.md
    - [ ] Run full test suite with coverage

- [ ] Task: Conductor - Automated Review 'Phase 3: Integration & Validation' (Protocol in workflow.md)
    - [ ] Run `/conductor:review` for Phase 3 changes
    - [ ] Apply Critical/High severity fixes automatically
    - [ ] Re-run tests after fixes
    - [ ] Commit review fixes
    - [ ] Push to remote and verify CI passes
    - [ ] Create checkpoint commit with git note

## Track Completion: Automated Final Review and Progression

- [ ] Task: Conductor - Final Track Review (Track Completion Protocol in workflow.md)
    - [ ] Invoke `/conductor:review` for entire track
    - [ ] Apply Critical/High/Medium severity fixes automatically
    - [ ] Run full test suite: `uv run pytest tests/ --cov=voiage --numprocesses=auto`
    - [ ] Commit review fixes
    - [ ] Push to remote and verify ALL CI workflows pass (retry up to 3 times)
    - [ ] Archive track to `conductor/archive/`
    - [ ] Update `conductor/tracks.md` to mark track as completed
    - [ ] Check for next track and auto-progress if exists
    - [ ] Final push and CI verification
