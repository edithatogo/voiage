# Migration Guide

This guide helps users migrate from other Value of Information tools to voiage, including BCEA, dampack, and voi.

## From BCEA (R)

### Key Differences

1. **Data Structures**: BCEA uses data frames, while voiage uses xarray-based data structures. `ValueArray` and `ParameterSet` keep xarray Datasets as the canonical in-memory representation.
2. **Function Names**: Function names differ between the two libraries
3. **Output Format**: voiage provides more structured output with better error handling

### Migration Example

#### BCEA (R)
```r
library(BCEA)

# Create BCEA object
bcea_obj <- bcea(eff, cost, ref = 1, interventions = c("Standard", "New"))

# Calculate EVPI
evpi_result <- evpi(bcea_obj)
```

#### voiage (Python)
```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

# Create net benefit data
net_benefits = cost - eff * k  # Adjust for willingness-to-pay threshold
value_array = ValueArray.from_numpy(net_benefits, ["Standard", "New"])

# Create DecisionAnalysis
analysis = DecisionAnalysis(nb_array=value_array)

# Calculate EVPI
evpi_result = analysis.evpi()
```

If you already have an xarray Dataset, use the dataset round-trip helpers
directly:

```python
from voiage.schema import ValueArray

value_array = ValueArray.from_dataset(dataset)
dataset_copy = value_array.to_dataset()
```

## From dampack (R)

### Key Differences

1. **VOI Methods**: dampack has extensive VOI functionality, much of which is replicated in voiage
2. **API Design**: voiage follows a more object-oriented approach
3. **Performance**: voiage is designed for better performance with large datasets

### Migration Example

#### dampack (R)
```r
library(dampack)

# Calculate EVPI
evpi_result <- calc_evpi(wtp, psa)

# Calculate EVPPI
evppi_result <- calc_evppi(wtp, psa, params)
```

#### voiage (Python)
```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet

# Create ValueArray and ParameterSet
value_array = ValueArray.from_numpy(net_benefits, ["Strategy 1", "Strategy 2"])
parameter_set = ParameterSet.from_numpy_or_dict(parameters)

# Create DecisionAnalysis
analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_set)

# Calculate EVPI
evpi_result = analysis.evpi()

# Calculate EVPPI
evppi_result = analysis.evppi()
```

If you are calling the lower-level `voiage.methods.basic.evppi` wrapper
directly, prefer `ParameterSet` inputs. Raw dict inputs still work as a
compatibility alias, but they now emit a deprecation warning.

For larger portable interchange artifacts, the core contract now treats JSON as
the committed fixture format and reserves Arrow/Parquet for binary exchange
paths where an optional backend is available.

## From voi (R)

### Key Differences

1. **Focus**: voi is specifically focused on VOI methods, similar to voiage
2. **Implementation**: voiage provides a more comprehensive implementation with better performance
3. **Extensibility**: voiage is designed to be more extensible with additional methods

### Migration Example

#### voi (R)
```r
library(voi)

# Calculate EVPI
evpi_result <- evpi(psa)

# Calculate EVPPI
evppi_result <- evppi(psa, params)
```

#### voiage (Python)
```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet

# Create ValueArray and ParameterSet
value_array = ValueArray.from_numpy(net_benefits)
parameter_set = ParameterSet.from_numpy_or_dict(parameters)

# Create DecisionAnalysis
analysis = DecisionAnalysis(nb_array=value_array, parameter_samples=parameter_set)

# Calculate EVPI
evpi_result = analysis.evpi()

# Calculate EVPPI
evppi_result = analysis.evppi()
```

## Feature Comparison

| Feature | BCEA | dampack | voi | voiage |
|---------|------|---------|-----|--------|
| EVPI | ✅ | ✅ | ✅ | ✅ |
| EVPPI | ✅ | ✅ | ✅ | ✅ |
| EVSI | ✅ | ✅ | ✅ | ✅ |
| ENBS | ❌ | ❌ | ✅ | ✅ |
| CEAF / dominance / heterogeneity | ❌ | Partial | ❌ | ✅ |
| Structural uncertainty VOI | ❌ | ❌ | ❌ | ✅ |
| Network Meta-Analysis VOI | ❌ | ❌ | ❌ | ✅ |
| Adaptive trials | ❌ | ❌ | ❌ | ✅ |
| Calibration VOI | ❌ | ❌ | ❌ | ✅ |
| Observational studies | ❌ | ❌ | ❌ | ✅ |
| Portfolio optimization | ❌ | ✅ | ❌ | ✅ |
| Sequential VOI | ❌ | ❌ | ❌ | ✅ |
| Python API | ❌ | ❌ | ❌ | ✅ |
| CLI workflow | ❌ | Partial | ❌ | ✅ |
| Cross-language scaffolds | ❌ | ❌ | ❌ | 🚧 |
| HEOML / ecosystem contracts | ❌ | ❌ | ❌ | 🚧 |
| Value of Perspective | ❌ | ❌ | ❌ | 🚧 |
| Distributional/equity VOI | ❌ | ❌ | ❌ | 🚧 |
| Implementation-adjusted VOI | ❌ | ❌ | ❌ | 🚧 |
| Frontier contract registry and validator | ❌ | ❌ | ❌ | ✅ |
| Preference-information / individualized-care VOI | ❌ | ❌ | Runtime + CLI + fixture-backed conformance | 🚧 |
| Validation, threshold, and robust VOI | ❌ | ❌ | Runtime + CLI + fixture-backed conformance | 🚧 |
| Causal, data-quality, computational, and elicitation VOI | ❌ | ❌ | ❌ | 📋 |
| Cross-domain usage | Limited | Limited | Limited | ✅ |

Current roadmap context:

- Core method work is complete.
- The active focus is spec-first expansion, conformance fixtures, cross-language binding scaffolds, and ecosystem contracts.
- The SOTA frontier track now includes implemented Value of Perspective,
  preference/individualized-care, model-validation, threshold/robust,
  distributional/equity, and implementation-adjusted APIs, along with a
  registry-backed frontier contract layer that validates the committed
  deterministic fixtures.
- Preference/individualized-care now has an implemented runtime surface, CLI
  entrypoint, and fixture-backed conformance contract; model-validation and
  threshold/robust also have runtime, CLI, and fixture-backed coverage.

## Best Practices for Migration

1. **Data Preparation**: Ensure your data is properly formatted for voiage's data structures
2. **Parameter Mapping**: Map parameters from your existing tools to voiage's parameter system
3. **Validation**: Validate results with known examples to ensure correct migration
4. **Performance Testing**: Test performance with your actual datasets
# Migration Guide

## Frontier maturity labels

Frontier method families use the governed maturity ladder in
the Astro frontier-governance developer guide. A family labelled
`fixture-backed` has deterministic repository fixtures, but it has not yet
passed cross-language parity and must not be described as stable. Stable
promotion additionally requires parity, documentation, changelog and
migration-guide evidence, and an explicit promotion approval.

Existing consumers should treat `method_maturity` as a capability statement,
not as a guarantee that external registries, hardware, or cloud runners have
been validated.
