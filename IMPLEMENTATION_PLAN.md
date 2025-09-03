# Implementation Plan for voiage

## Current Status

As of Q3 2025, the voiage library has made significant progress:
- Core infrastructure is fully implemented
- Basic VOI methods (EVPI, EVPPI) are complete
- Portfolio optimization is fully implemented
- Plotting capabilities are partially implemented
- EVSI implementation is in progress but needs completion

## Immediate Priorities (Next 2-4 Weeks)

### 1. Complete EVSI Implementation
**Status**: In Progress
**Owner**: TBD
**Estimated Effort**: 2-3 weeks

#### Tasks:
- [ ] Activate and fix tests in `tests/test_sample_information.py`
  - Uncomment existing test functions
  - Fix any issues preventing tests from passing
  - Add additional test cases for edge conditions
- [ ] Implement regression-based method for EVSI
  - Add regression-based EVSI calculation method
  - Implement proper metamodeling approaches
  - Ensure compatibility with existing API
- [ ] Improve the `two_loop` implementation
  - Optimize performance of the two-loop Monte Carlo method
  - Add better error handling and validation
  - Improve documentation and examples

#### Deliverables:
- Fully functional EVSI implementation with both two-loop and regression methods
- Comprehensive test suite for EVSI functions
- Documentation and examples for EVSI usage

### 2. Finalize Data Structure Transition
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: 1 week

#### Tasks:
- [ ] Remove `voiage.core.data_structures` module
- [ ] Replace all internal usages with `voiage.schema`
- [ ] Update `DecisionAnalysis` and method signatures to use `ParameterSet` and `ValueArray` directly
- [ ] Update all tests to use new data structures
- [ ] Update documentation and examples

#### Deliverables:
- Clean codebase with consistent data structure usage
- Updated documentation reflecting the new API
- All tests passing with new data structures

### 3. Enhance Plotting Module
**Status**: In Progress
**Owner**: TBD
**Estimated Effort**: 1-2 weeks

#### Tasks:
- [ ] Complete implementation of CEAC plotting functionality in `voiage/plot/ceac.py`
  - Implement all required plotting functions
  - Add comprehensive test coverage
  - Improve documentation and examples
- [ ] Expand VOI curve plotting capabilities
  - Add more plotting options and customization
  - Improve error handling and validation
- [ ] Add comprehensive examples and documentation
  - Create detailed usage examples
  - Add API documentation for all plotting functions

#### Deliverables:
- Fully functional plotting module with CEAC and VOI curve plotting
- Comprehensive test suite for plotting functions
- Detailed documentation and examples

## Short-term Goals (Next 2-3 Months)

### 1. Validation & Benchmarking
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: 3-4 weeks

#### Tasks:
- [ ] Create validation notebooks replicating results from established R packages
  - Replicate examples from BCEA package
  - Replicate examples from voi package
  - Validate results against published studies
- [ ] Benchmark performance of current implementations
  - Compare performance against R implementations
  - Identify bottlenecks and optimization opportunities
  - Document performance characteristics

#### Deliverables:
- Validation notebooks demonstrating correctness
- Performance benchmarking report
- Documentation of validation results

### 2. Network Meta-Analysis VOI Implementation
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: 4-6 weeks

#### Tasks:
- [ ] Define required data structures in `voiage.schema`
  - Add multivariate parameter distributions
  - Add NMA-specific data structures
- [ ] Implement `evsi_nma` function in `voiage/methods/network_nma.py`
  - Implement core NMA VOI calculation
  - Add comprehensive test coverage
  - Document usage and examples

#### Deliverables:
- Fully functional NMA VOI implementation
- Comprehensive test suite
- Documentation and examples

### 3. Advanced Method Implementation
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: 6-8 weeks

#### Tasks:
- [ ] Implement Structural VOI methods in `voiage/methods/structural.py`
  - Complete placeholder implementation
  - Add comprehensive test coverage
  - Document usage and examples
- [ ] Implement Sequential VOI functionality in `voiage/methods/sequential.py`
  - Complete placeholder implementation
  - Add comprehensive test coverage
  - Document usage and examples

#### Deliverables:
- Fully functional Structural and Sequential VOI implementations
- Comprehensive test suites
- Documentation and examples

## Medium-term Goals (Next 6-12 Months)

### 1. Cross-Domain Expansion
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: 8-12 weeks

#### Tasks:
- [ ] Develop cross-domain examples for business and environmental applications
  - Create detailed tutorial notebooks
  - Validate with domain experts
  - Document cross-domain usage patterns
- [ ] Enhance documentation for cross-domain usage
  - Add cross-domain examples to main documentation
  - Create domain-specific guides

#### Deliverables:
- Cross-domain tutorial notebooks
- Enhanced documentation for cross-domain usage
- Validation with domain experts

### 2. Performance Optimization
**Status**: In Progress
**Owner**: TBD
**Estimated Effort**: 6-8 weeks

#### Tasks:
- [ ] Optimize JAX backend implementation
  - Improve performance of JAX-based calculations
  - Add more comprehensive JAX support
  - Document performance benefits
- [ ] Implement additional metamodels in `voiage/metamodels.py`
  - Add Gaussian Process metamodels
  - Add other advanced metamodeling approaches
  - Document usage and performance characteristics

#### Deliverables:
- Optimized JAX backend implementation
- Additional metamodeling capabilities
- Performance documentation

### 3. Quality Assurance
**Status**: Ongoing
**Owner**: TBD
**Estimated Effort**: 4-6 weeks

#### Tasks:
- [ ] Achieve 90%+ test coverage
  - Add tests for currently untested code paths
  - Improve existing test quality
  - Add property-based testing with Hypothesis
- [ ] Implement comprehensive integration tests
  - Add end-to-end integration tests
  - Test various usage scenarios
  - Validate cross-module interactions

#### Deliverables:
- 90%+ test coverage across the codebase
- Comprehensive integration test suite
- Property-based tests for key functionality

## Long-term Vision (12+ Months)

### 1. Ecosystem Development
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: Ongoing

#### Tasks:
- [ ] Establish language-agnostic API specification
  - Define JSON Schema for inputs/outputs
  - Document API specification
  - Create validation tools
- [ ] Begin planning for R and Julia ports
  - Create prototype implementations
  - Validate cross-language design
  - Document porting process

#### Deliverables:
- Language-agnostic API specification
- Prototype R and Julia implementations
- Documentation for porting process

### 2. Advanced Capabilities
**Status**: To Do
**Owner**: TBD
**Estimated Effort**: Ongoing

#### Tasks:
- [ ] Implement machine learning-based metamodels
  - Add neural network metamodels
  - Add other ML-based approaches
  - Document performance and accuracy
- [ ] Add support for real-time VOI calculations
  - Implement streaming data support
  - Add real-time calculation capabilities
  - Document usage patterns

#### Deliverables:
- ML-based metamodeling capabilities
- Real-time VOI calculation support
- Documentation and examples

## Resource Requirements

### Personnel
- 2-3 core developers for immediate priorities
- 1-2 domain experts for validation
- 1 documentation specialist

### Tools & Infrastructure
- CI/CD pipeline (already in place)
- Testing infrastructure (already in place)
- Documentation system (Sphinx/ReadTheDocs)
- Performance profiling tools

### Timeline
- Immediate priorities: 2-4 weeks
- Short-term goals: 2-3 months
- Medium-term goals: 6-12 months
- Long-term vision: 12+ months

## Success Metrics

### Code Quality
- Test coverage > 90%
- Code linting and formatting compliance
- Type hinting coverage > 95%
- Documentation coverage > 90%

### Performance
- Benchmark performance against R implementations
- Response time for core calculations < 1 second for typical use cases
- Memory usage optimization

### Community Engagement
- Number of contributors
- Issue response time
- Documentation quality scores
- User feedback and adoption metrics

## Risk Mitigation

### Technical Risks
- Complexity of advanced VOI methods: Mitigate through incremental implementation and thorough testing
- Performance bottlenecks: Mitigate through profiling and optimization
- Integration challenges: Mitigate through comprehensive integration testing

### Resource Risks
- Developer availability: Mitigate through community engagement and clear contribution guidelines
- Domain expertise: Mitigate through collaboration with domain experts
- Infrastructure costs: Mitigate through use of open-source tools and cloud credits

### Timeline Risks
- Scope creep: Mitigate through clear milestone definitions and regular progress reviews
- Technical blockers: Mitigate through early identification and alternative approaches
- Dependency issues: Mitigate through careful dependency management and version pinning