# Network Meta-Analysis VOI

This guide covers the Network Meta-Analysis (NMA) Value of Information functionality in voiage.

## Overview

Network Meta-Analysis VOI methods extend traditional VOI analysis to compare multiple interventions simultaneously within a network structure.

## Key Concepts

### Model Structure
NMA VOI analysis involves:
- Multiple treatment arms in a network
- Comparative effectiveness data
- Uncertainty in relative treatment effects

## Usage Example

```python
import numpy as np
from voiage.methods.network_nma import nma_evpi, nma_evppi

# Example NMA data structure
# This is a simplified example - real NMA would have more complex data
treatment_effects = {
    "A vs B": np.random.normal(0.1, 0.05, 1000),
    "A vs C": np.random.normal(0.2, 0.05, 1000),
    "B vs C": np.random.normal(0.1, 0.05, 1000)
}

# Calculate NMA EVPI
evpi_result = nma_evpi(treatment_effects)
print(f"NMA EVPI: {evpi_result:.2f}")

# Calculate NMA EVPPI for specific comparisons
evppi_result = nma_evppi(treatment_effects, parameters_of_interest=["A vs B"])
print(f"NMA EVPPI for A vs B: {evppi_result:.2f}")
```

## Data Structure

NMA VOI methods expect data in the form of treatment effect comparisons with their uncertainties.

## Best Practices

1. **Network Consistency**: Ensure the network of treatment comparisons is consistent
2. **Uncertainty Representation**: Properly represent uncertainty in treatment effects
3. **Clinical Relevance**: Focus on clinically relevant treatment comparisons
4. **Model Validation**: Validate results with clinical experts