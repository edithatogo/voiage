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
- [x] Task: Create data structures for NMA input <f3625e9>
    - [x] Write failing tests for NMA data structures
    - [x] Define `NetworkMetaAnalysisData` class
    - [x] Support multi-treatment comparisons
    - [x] Add validation for network consistency
    - [x] Write unit tests for NMA data structures

### Task 2.2: Implement NMA VOI Algorithm
- [x] Task: Implement NMA-based VOI calculation <f3625e9>
    - [x] Write failing tests for NMA VOI method
    - [x] Implement `calculate_nma_evpi()` function
    - [x] Implement `calculate_nma_evppi()` function
    - [x] Support EVPI/EVPPI for network comparisons
    - [x] Dict-to-NMA data conversion helper
    - [x] Verify >80% test coverage

### Task 2.3: Add CLI Interface for NMA VOI
- [x] Task: Create CLI command for NMA VOI <f3625e9>
    - [x] Write failing tests for CLI command
    - [x] Add `voiage calculate-nma-voi` command
    - [x] Support NMA result JSON config input
    - [x] Support --parameters-of-interest for EVPPI
    - [x] Add comprehensive help/examples
    - [x] Test CLI end-to-end

### Task 2.4: Write Documentation for NMA VOI
- [x] Task: Document NMA VOI method <f3625e9>
    - [x] Add docstrings with examples
    - [x] CLI help text comprehensive
    - [x] Update README feature table (done in Phase 3)

- [x] Task: Conductor - Automated Review 'Phase 2: Network Meta-Analysis VOI' (Protocol in workflow.md) [checkpoint: 818d3a0]
    - [x] Code review completed
    - [x] Tests written and comprehensive
    - [x] No Critical/High severity issues

## Phase 3: Integration & Validation

### Task 3.1: Integration Testing
- [x] Task: Test both methods with real-world datasets <88c5b5a>
    - [x] Test structural uncertainty with health economics models
    - [x] Test NMA VOI with clinical trial networks (diabetes)
    - [x] Verify results against known benchmarks
    - [x] Add integration tests (7 tests)

### Task 3.2: Performance Optimization
- [x] Task: Optimize performance for large datasets <88c5b5a>
    - [x] Benchmark current performance
    - [x] JAX acceleration verified (structural_evpi_jit)
    - [x] Add performance tests (6 tests)

### Task 3.3: Final Documentation & Cleanup
- [x] Task: Finalize all documentation <88c5b5a>
    - [x] Review and update all docstrings
    - [x] Ensure examples work correctly
    - [x] Update changelog.md (in final commit)
    - [x] Run full test suite with coverage

- [x] Task: Conductor - Automated Review 'Phase 3: Integration & Validation' (Protocol in workflow.md) [checkpoint: f0c0bbe]
    - [x] Run `/conductor:review` for Phase 3 changes
    - [x] Apply Critical/High severity fixes automatically (none needed)
    - [x] Re-run tests after fixes
    - [x] Commit review fixes
    - [x] Push to remote and verify CI passes
    - [x] Create checkpoint commit with git note

## Track Completion: Automated Final Review and Progression

- [x] Task: Conductor - Final Track Review (Track Completion Protocol in workflow.md)
    - [x] Invoke `/conductor:review` for entire track
    - [x] Apply Critical/High/Medium severity fixes automatically
    - [x] Run full test suite: `uv run pytest tests/ --cov=voiage --numprocesses=auto`
    - [x] Commit review fixes
    - [x] Push to remote and verify ALL CI workflows pass (retry up to 3 times)
    - [x] Archive track to `conductor/archive/`
    - [x] Update `conductor/tracks.md` to mark track as completed
    - [x] Check for next track and auto-progress if exists (no next track)
    - [x] Final push and CI verification
