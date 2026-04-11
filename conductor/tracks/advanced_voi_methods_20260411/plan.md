# Implementation Plan: Advanced VOI Methods

## Phase 1: Structural Uncertainty VOI

### Task 1.1: Define Data Structures for Structural Uncertainty
- [ ] Task: Create data structures for multiple model structures
    - [ ] Define `ModelStructure` class with inputs/outputs
    - [ ] Define `StructuralUncertaintyData` container
    - [ ] Add validation for model structure compatibility
    - [ ] Write unit tests for data structures

### Task 1.2: Implement Structural Uncertainty VOI Algorithm
- [ ] Task: Implement core structural uncertainty VOI calculation
    - [ ] Write failing tests for structural VOI method
    - [ ] Implement `calculate_structural_voi()` function
    - [ ] Support model averaging approaches
    - [ ] Add JAX-accelerated backend
    - [ ] Verify >80% test coverage

### Task 1.3: Add CLI Interface for Structural Uncertainty
- [ ] Task: Create CLI command for structural uncertainty VOI
    - [ ] Add `voiage calculate-structural-voi` command
    - [ ] Support CSV input format
    - [ ] Add comprehensive help/examples
    - [ ] Test CLI end-to-end

### Task 1.4: Write Documentation for Structural Uncertainty
- [ ] Task: Document structural uncertainty VOI method
    - [ ] Add docstrings with examples
    - [ ] Create Jupyter notebook example
    - [ ] Update README feature table

- [ ] Task: Conductor - Automated Review 'Phase 1: Structural Uncertainty VOI' (Protocol in workflow.md)
    - [ ] Run `/conductor:review` for Phase 1 changes
    - [ ] Apply Critical/High severity fixes automatically
    - [ ] Re-run tests after fixes
    - [ ] Commit review fixes
    - [ ] Push to remote and verify CI passes
    - [ ] Create checkpoint commit with git note

## Phase 2: Network Meta-Analysis VOI

### Task 2.1: Define Data Structures for Network Meta-Analysis
- [ ] Task: Create data structures for NMA input
    - [ ] Define `NetworkMetaAnalysisData` class
    - [ ] Support multi-treatment comparisons
    - [ ] Add validation for network consistency
    - [ ] Write unit tests for NMA data structures

### Task 2.2: Implement NMA VOI Algorithm
- [ ] Task: Implement NMA-based VOI calculation
    - [ ] Write failing tests for NMA VOI method
    - [ ] Implement `calculate_nma_voi()` function
    - [ ] Support EVPI/EVPPI for network comparisons
    - [ ] Add JAX-accelerated backend
    - [ ] Verify >80% test coverage

### Task 2.3: Add CLI Interface for NMA VOI
- [ ] Task: Create CLI command for NMA VOI
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
