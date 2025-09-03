# Migration Guide

This guide helps users migrate from other Value of Information tools to voiage, including BCEA, dampack, and voi.

## From BCEA (R)

### Key Differences

1. **Data Structures**: BCEA uses data frames, while voiage uses xarray-based data structures
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
| Portfolio Optimization | ❌ | ✅ | ❌ | ✅ |
| Network Meta-Analysis | ❌ | ❌ | ❌ | ✅ |
| Adaptive Trials | ❌ | ❌ | ❌ | ✅ |
| Calibration | ❌ | ❌ | ❌ | ✅ |
| Observational Studies | ❌ | ❌ | ❌ | ✅ |
| Python API | ❌ | ❌ | ❌ | ✅ |
| Cross-Domain Usage | Limited | Limited | Limited | ✅ |

## Best Practices for Migration

1. **Data Preparation**: Ensure your data is properly formatted for voiage's data structures
2. **Parameter Mapping**: Map parameters from your existing tools to voiage's parameter system
3. **Validation**: Validate results with known examples to ensure correct migration
4. **Performance Testing**: Test performance with your actual datasets