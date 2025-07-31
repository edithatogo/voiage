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

## Phase 2: JAX-Native High-Performance Core
1. **JAX Backend for Core Numerics**
    - Refactor core VOI functions in `voiage.methods` to use `jax.numpy` and `jax.scipy` instead of `numpy` and `scipy`.
    - Apply `@jax.jit` to these functions to enable XLA compilation for significant performance gains.
    - Update the `voiage/backends.py` module to manage the JAX backend as the primary high-performance option.
2. **JAX-Native Metamodels**
    - Replace the planned `scikit-learn` and `statsmodels` based metamodels (`GAMMetamodel`, `GPMetamodel`) with JAX-native equivalents.
    - Implement a `FlaxMetamodel` or `HaikuMetamodel` for neural network-based metamodeling.
    - Implement a `tinygpMetamodel` for lightweight Gaussian Process regression, ensuring the entire EVSI calculation can be JIT-compiled.
3. **XArray with JAX Backend Integration**
    - Ensure that `ValueArray` and `ParameterSet` data structures internally use `jax.Array` instances when the JAX backend is active.
    - Minimize data transfers between CPU (NumPy) and accelerator (JAX) by keeping the dataflow within the JAX ecosystem.
4. **Update `DecisionAnalysis` Class**
    - Refactor the `DecisionAnalysis` class to handle `jax.Array` inputs seamlessly and dispatch calculations to the JIT-compiled functions.

## Phase 3: State-of-the-Art Health Economics Core
1. **Robust EVSI**
   - Refactor `evsi` into `DecisionAnalysis`.
   - Implement two-loop Monte Carlo with Bayesian updaters (`ConjugateUpdater`, `NumPyroUpdater`) and JAX-native metamodels.
2. **Network Meta-Analysis**
   - Design data structures for network evidence.
   - Add an `evsi_nma` method supporting multivariate posteriors.
3. **Validation & Benchmarking**
   - Add notebooks in `validation/` replicating published results (e.g., R `BCEA`).
   - Benchmark `numpy` vs `jax` backends.
4. **Plotting & Examples**
   - Implement plotting utilities (CEAC, VOI curves, EVPPI surfaces).
   - Create a tutorial notebook under `examples/health_economics`.

## Phase 4: Advanced Methods & Cross-Domain Expansion
1. **Advanced VOI methods**
   - Implement `portfolio_voi`, `structural_voi` and `sequential_voi` methods using the OO API.
2. **Cross-domain examples**
   - Provide notebooks for environmental policy and business strategy applications.
3. **XArray integration**
   - Refactor `ParameterSet` and `ValueArray` to leverage `xarray.Dataset` with labelled dimensions throughout the codebase.

## Phase 5: Ecosystem & Future Ports
1. **Community growth**
   - Maintain issues and discussions, publish periodic updates and present at conferences.
2. **Language-agnostic specification**
   - Draft a JSON Schema for `DecisionAnalysis` inputs/outputs to guide R and Julia ports.
3. **Prototype ports**
   - Begin experimental implementations of `rvoiage` (R) and `Voiage.jl` (Julia) to validate the cross-language design.

This plan should be revisited regularly as features land to keep the roadmap and to-do list in sync.
