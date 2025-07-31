# voiage To-Do List (v2)

This to-do list is based on the revised project roadmap and incorporates feedback from the expert reviews.

---

## Phase 1: Foundation & API Refactoring

- [ ] **1. API & Core Structure**
    - [ ] Design `DecisionAnalysis` class API in a new `voiage/analysis.py` module.
    - [ ] Refactor `evpi()` and `evppi()` from `methods/basic.py` to be methods of the `DecisionAnalysis` class.
    - [ ] Move existing data structures from `core/data_structures.py` into a new `voiage/schema.py` or similar.
    - [ ] Rename `PSASample` to `ParameterSet`.
    - [ ] Rename `TrialArm` to `DecisionOption`.
    - [ ] Rename `NetBenefitArray` to `ValueArray`.
    - [ ] Update all existing code and tests to use the new class-based API and renamed data structures.
    - [ ] Add lightweight functional wrappers (e.g., `voiage.evpi()`) that use the core OO-API internally.

- [ ] **2. High-Performance Backend**
    - [ ] Create a `voiage/backends.py` module to manage computational backends.
    - [ ] Implement a JAX version of the core VOI calculations.
    - [ ] Add a global `voiage.set_backend('numpy' | 'jax')` function.
    - [ ] Create initial performance benchmarks comparing the two backends.

- [ ] **3. Project Infrastructure & Documentation**
    - [ ] Configure GitHub Actions workflow for CI (pytest, coverage, linting).
    - [ ] Set up a Sphinx project in the `docs/` directory.
    - [ ] Choose and configure a modern theme (e.g., Furo, PyData Sphinx Theme).
    - [ ] Write the initial "Getting Started" guide based on the new OO-API.
    - [ ] Set up GitHub Pages to auto-deploy the documentation on pushes to `main`.
    - [ ] Create the `CONTRIBUTING.md` file.
    - [ ] Create a `CODE_OF_CONDUCT.md` file.
    - [ ] Move all test logic from `if __name__ == "__main__"` blocks into the `tests/` directory.

---

## Phase 2: JAX-Native High-Performance Core

- [ ] **1. JAX Backend for Core Numerics**
    - [ ] Refactor `voiage.methods.basic.evpi` to use `jax.numpy` and `@jax.jit`.
    - [ ] Refactor `voiage.methods.basic.evppi` to use `jax.numpy` and `@jax.jit`.
    - [ ] Update `voiage/backends.py` to make JAX the default high-performance backend.
    - [ ] Expand performance benchmarks to compare `numpy` vs. `jax` on core functions.

- [ ] **2. JAX-Native Metamodels**
    - [ ] Design a `JAXMetamodel` protocol/base class.
    - [ ] Implement a `FlaxMetamodel` for EVSI, replacing the planned `GAMMetamodel`.
    - [ ] Implement a `tinygpMetamodel` for EVSI, replacing the planned `GPMetamodel`.

- [ ] **3. XArray Backend Integration**
    - [ ] Add logic to `ValueArray` and `ParameterSet` to wrap `jax.Array` instances when the JAX backend is enabled.
    - [ ] Write tests to verify that data remains on the JAX device during a full analysis pipeline.

- [ ] **4. Update `DecisionAnalysis` Class**
    - [ ] Modify `DecisionAnalysis.__init__` to accept `jax.Array` inputs.
    - [ ] Ensure `evpi()` and `evppi()` methods correctly dispatch to JIT-compiled functions.

---

## Phase 3: State-of-the-Art Health Economics Core

- [ ] **1. EVSI Implementation**
    - [ ] Refactor `evsi()` to be a method of the `DecisionAnalysis` class.
    - [ ] Implement a full two-loop Monte Carlo algorithm.
    - [ ] Add a `BayesianUpdater` protocol/class that can be passed to the `evsi` method.
    - [ ] Implement a `ConjugateUpdater` for common conjugate prior models (e.g., Normal-Normal).
    - [ ] Create a `NumPyroUpdater` that can take a user-defined NumPyro model, run MCMC, and return the posterior.
    - [ ] Add a `Metamodel` protocol/class.
    - [ ] Implement a `GAMMetamodel` and `GPMetamodel` using `scikit-learn` or other libraries.

- [ ] **2. NMA VOI**
    - [ ] Design data structures to represent network evidence (e.g., relative effects).
    - [ ] Implement `evsi_nma()` as a method, capable of handling multivariate posteriors from an NMA.

- [ ] **3. Validation, Plotting, & Examples**
    - [ ] Create a `validation/` directory with notebooks.
    - [ ] Add a validation notebook replicating results from the R `BCEA` package.
    - [ ] Implement `analysis.plot_ceac()`.
    - [ ] Implement `analysis.plot_evppi_curve()`.
    - [ ] Create `examples/health_economics/01_basic_voi` notebook with synthetic data.

---

## Phase 4: Advanced Methods & Cross-Domain Expansion

- [ ] **1. Advanced Methods**
    - [ ] Implement `analysis.portfolio_voi()`.
    - [ ] Implement `analysis.structural_voi()`.
    - [ ] Implement `analysis.sequential_voi()`.

- [ ] **2. Cross-Domain Examples**
    - [ ] Create `examples/environmental_policy/01_basic_voi` notebook with synthetic data.
    - [ ] Create `examples/business_strategy/01_portfolio_optimization` notebook with synthetic data.

- [ ] **3. XArray Integration**
    - [ ] Begin refactoring `ParameterSet` and `ValueArray` to be based on `xarray.Dataset`, providing labeled dimensions.

---

## Phase 5: Ecosystem & Future Ports (Backlog)

- [ ] **1. Community**
    - [ ] Create "Good First Issue" and "Help Wanted" issue templates on GitHub.
    - [ ] Publish a blog post announcing `voiage` v1.0.

- [ ] **2. API Specification**
    - [ ] Draft a formal JSON Schema representation of the `DecisionAnalysis` inputs and outputs.

- [ ] **3. R & Julia Ports**
    - [ ] Begin experimental prototype of an R6-based `rvoiage` package.
    - [ ] Begin experimental prototype of a `Voiage.jl` module.
