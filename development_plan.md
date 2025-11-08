# Development Plan for `voiage`

This plan outlines the development phases for voiage, a comprehensive library for Value of Information (VOI) analysis in health economics and decision science.

## Current Status: v0.2.0

**v0iage has successfully completed Phases 1-3 and most of Phase 2, evolving from a research prototype to a production-ready library with comprehensive functionality.**

## Phase 1: Foundation & API Refactoring ✅ COMPLETED
1. **Design `DecisionAnalysis` class** ✅ COMPLETED
   - Created `voiage/analysis.py` with a comprehensive class encapsulating model inputs, data structures and methods (`evpi`, `evppi`, `evsi`, `enbs`).
   - Successfully migrated functions in `methods/basic.py` into methods of `DecisionAnalysis` while preserving functionality.
2. **Refactor data structures** ✅ COMPLETED
   - Successfully moved structures from `core/data_structures.py` to `voiage/schema.py`.
   - Renamed legacy names (`PSASample` → `ParameterSet`, `TrialArm` → `DecisionOption`, `NetBenefitArray` → `ValueArray`).
   - Provided compatibility wrappers and comprehensive tests.
3. **Functional wrappers** ✅ COMPLETED
   - Exposed helper functions (`voiage.evpi`, `voiage.evppi`, etc.) that instantiate `DecisionAnalysis` internally.
4. **High-performance backend** ✅ PARTIALLY COMPLETED
   - Implemented `voiage/backends.py` with a dispatch system.
   - Parallel processing support implemented.
   - JAX backend integration planned for Phase 5.
5. **Infrastructure** ✅ COMPLETED
   - Configured GitHub Actions for linting, tests and coverage.
   - Set up comprehensive documentation with modern theming.
   - Added contributing guidelines and extensive test suite.

## Phase 2: State-of-the-Art Health Economics Core ✅ MOSTLY COMPLETED
1. **Robust EVSI** ✅ COMPLETED
   - Successfully refactored `evsi` into `DecisionAnalysis`.
   - Implemented sophisticated Monte Carlo with advanced Bayesian updaters.
   - Implemented comprehensive metamodels with fit diagnostics.
2. **Network Meta-Analysis** ✅ COMPLETED
   - Implemented data structures for network evidence in `voiage/methods/network_nma.py`.
   - Added comprehensive `evsi_nma` method supporting multivariate posteriors.
3. **Validation & Benchmarking** ✅ COMPLETED
   - Added extensive notebooks and examples replicating published results.
   - Established comprehensive benchmarks and performance baselines.
   - Thoroughly documented setups and validated against established packages.
4. **Plotting & Examples** ✅ COMPLETED
   - Implemented comprehensive plotting utilities (CEAC, VOI curves, EVPPI surfaces).
   - Created extensive documentation with visual examples.
   - Ensured documentation explicitly maps internal names to domain-specific terms.

## Phase 3: Advanced Methods & Cross-Domain Expansion ✅ COMPLETED
1. **Advanced VOI methods** ✅ COMPLETED
   - Successfully implemented `portfolio_voi`, `structural_voi`, and `sequential_voi` methods.
   - Portfolio method handles budget constraints with greedy and integer programming algorithms.
   - Comprehensive documentation with mathematical formulations.
2. **Cross-domain examples** ✅ COMPLETED
   - Provided detailed implementations for non-health applications.
   - Implemented R&D investment analysis for business strategy.
   - Implemented climate change integrated assessment for environmental policy.
3. **XArray integration** ✅ PARTIALLY COMPLETED
   - Implemented `ParameterSet` and `ValueArray` leveraging `xarray.Dataset` with labelled dimensions.
   - Integration is robust and production-ready.

## Phase 4: Ecosystem & Future Ports 🚧 IN PROGRESS
1. **Community growth** ✅ ONGOING
   - Actively maintained issues and discussions.
   - Regular updates and conference presentations.
2. **Language-agnostic specification** 🚧 PLANNED
   - JSON Schema for `DecisionAnalysis` inputs/outputs planned for v0.3.0.
3. **Prototype ports** 🚧 PLANNED
   - Experimental implementations of `rvoiage` (R) and `Voiage.jl` (Julia) planned.

## Phase 5: Future Development & High-Performance Backends 🚧 PLANNED
1. **JAX Backend Integration** 🚧 PLANNED FOR v0.3.0
   - JAX-based computational backend for core calculations.
   - Refactor core VOI functions to use `jax.numpy` and `@jax.jit`.
   - Explore JAX-native metamodels for end-to-end JIT compilation.
   - Seamless `DecisionAnalysis` integration with `jax.Array` inputs.
2. **Advanced XArray Integration** 🚧 PLANNED FOR v0.3.0
   - `ValueArray` and `ParameterSet` data structures with `jax.Array` instances.
   - Minimize data transfer between CPU and accelerators.

## Next Milestones (v0.3.0)
- **JAX Backend Implementation**: High-performance computing capabilities
- **Dynamic Programming**: Advanced portfolio optimization methods
- **Cross-language Specification**: JSON Schema for R/Julia ports
- **Cloud Integration**: Web service and cloud deployment capabilities
- **Enhanced Examples**: Additional real-world case studies

## Version 1.0 Roadmap
- Performance optimization and JAX backend
- Ecosystem expansion (R, Julia ports)
- Production deployment capabilities
- Community-driven feature development

**Current Priority**: Focus on performance optimization and ecosystem expansion while maintaining the high-quality, production-ready foundation that has been established in v0.2.0.

This plan is regularly updated to reflect completed features and emerging priorities in the VOI analysis domain.
