# Track Specification: Advanced VOI Methods

## Overview
This track implements two critical advanced VOI methods as outlined in the v0.4 roadmap:
1. **Structural Uncertainty VOI** - For comparing different model structures
2. **Network Meta-Analysis VOI** - For synthesizing evidence from multiple studies

## Background
Current voiage implementation covers core methods (EVPI, EVPPI, EVSI, ENBS) but lacks support for advanced scenarios where:
- Multiple model structures compete (structural uncertainty)
- Evidence from multiple studies must be synthesized (network meta-analysis)

These methods are not available in any competing R or commercial tools, making this a key differentiator for voiage.

## Requirements

### Functional Requirements
1. **Structural Uncertainty VOI**
   - Accept multiple model structures as input
   - Calculate VOI accounting for structural uncertainty
   - Support model averaging approaches
   - Provide CLI interface

2. **Network Meta-Analysis VOI**
   - Accept network meta-analysis results
   - Calculate EVPI/EVPPI for network comparisons
   - Support multi-parameter evidence synthesis
   - Provide CLI interface

### Non-Functional Requirements
- Performance: Handle 10,000+ PSA simulations efficiently
- JAX acceleration supported
- Full test coverage (>80%)
- Comprehensive documentation

## Acceptance Criteria
- [ ] Structural uncertainty VOI method implemented and tested
- [ ] Network meta-analysis VOI method implemented and tested
- [ ] CLI commands available for both methods
- [ ] Documentation updated with examples
- [ ] All tests passing with >80% coverage
