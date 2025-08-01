# Development Plan for `voiage`

This plan outlines the steps to implement the remaining features from `roadmap.md` and `todo.md`.

## Phase 1: Foundation & API Refactoring
1. **Design `DecisionAnalysis` class**
   - Create `voiage/analysis.py` with a class encapsulating model inputs, data structures and methods (`evpi`, `evppi`, `evsi`).
   - Migrate functions in `methods/basic.py` into methods of `DecisionAnalysis` while preserving their functionality.
2. **Refactor data structures**
   - Move structures from `core/data_structures.py` to a new schema module.
   - Rename legacy names (`PSASample` → `ParameterSet`, `TrialArm` → `DecisionOption`, `NetBenefitArray` → `ValueArray`).
   - Provide lightweight compatibility wrappers and update tests.
3. **Functional wrappers**
   - Expose helper functions (`voiage.evpi`, `voiage.evppi`, etc.) that instantiate `DecisionAnalysis` internally.
4. **High-performance backend**
   - Implement `voiage/backends.py` with a dispatch system and a JAX backend for core calculations.
5. **Infrastructure**
   - Configure GitHub Actions for linting, tests and coverage.
   - Set up Sphinx docs with a modern theme and deploy via GitHub Pages.
   - Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and move inline test code into `tests/`.

## Phase 2: State-of-the-Art Health Economics Core
1. **Robust EVSI**
   - Refactor `evsi` into `DecisionAnalysis`.
   - Implement two-loop Monte Carlo with Bayesian updaters (`ConjugateUpdater`, `NumPyroUpdater`).
   - Implement JAX-native metamodels (`GAMMetamodel`, `GPMetamodel`) and include metamodel fit diagnostics.
2. **Network Meta-Analysis**
   - Design data structures for network evidence.
   - Add an `evsi_nma` method supporting multivariate posteriors. This is a key feature for HTA.
3. **Validation & Benchmarking**
   - Add notebooks in `validation/` replicating published results (e.g., R `BCEA`). Meticulously document setups and explain any discrepancies.
   - Benchmark the `numpy` backend to establish a performance baseline.
4. **Plotting & Examples**
   - Implement plotting utilities (CEAC, VOI curves, EVPPI surfaces).
   - Create a tutorial notebook under `examples/health_economics`.
   - Ensure documentation and examples explicitly map internal names (e.g., `DecisionOption`) to domain-specific terms (e.g., `TrialArm`).

## Phase 3: Advanced Methods & Cross-Domain Expansion
1. **Advanced VOI methods**
   - Note: These methods require a dedicated research and design sub-phase before implementation. This includes writing out the mathematical formulation in the documentation.
   - Implement `portfolio_voi`, `structural_voi` and `sequential_voi` methods using the OO API. The portfolio method must handle budget constraints and explore different algorithms (e.g., greedy vs. integer programming).
2. **Cross-domain examples**
   - Provide detailed notebooks for non-health applications, such as an R&D investment problem for business strategy and a climate change integrated assessment model for environmental policy.
3. **XArray integration**
   - Refactor `ParameterSet` and `ValueArray` to leverage `xarray.Dataset` with labelled dimensions throughout the codebase.

## Phase 4: Ecosystem & Future Ports
1. **Community growth**
   - Maintain issues and discussions, publish periodic updates and present at conferences.
2. **Language-agnostic specification**
   - Draft a JSON Schema for `DecisionAnalysis` inputs/outputs to guide R and Julia ports.
3. **Prototype ports**
   - Begin experimental implementations of `rvoiage` (R) and `Voiage.jl` (Julia) to validate the cross-language design.

## Phase 5: Future Development & High-Performance Backends
1. **JAX Backend Integration**
    - Re-evaluate and implement a JAX-based computational backend for core calculations.
    - Refactor core VOI functions to use `jax.numpy` and `@jax.jit`.
    - Explore JAX-native metamodels (e.g., `tinygp`, `Flax`) for end-to-end JIT compilation.
    - Update the `DecisionAnalysis` class to seamlessly handle `jax.Array` inputs and dispatch to JIT-compiled functions.
2. **Advanced XArray Integration**
    - Ensure that `ValueArray` and `ParameterSet` data structures can internally use `jax.Array` instances when the JAX backend is active, minimizing data transfer between CPU and accelerators.

This plan should be revisited regularly as features land to keep the roadmap and to-do list in sync.
