# Advanced VOI Methods

This guide covers the advanced Value of Information methods implemented in voiage, including adaptive trials, calibration, and observational studies.

## Adaptive Trial VOI

Adaptive trial VOI methods evaluate the value of information from clinical trials with interim analyses and potential modifications.

### Usage Example

```python
import numpy as np
from voiage.methods.adaptive import adaptive_evsi

# Define trial design parameters
sample_size = 1000
interim_analysis_times = [0.5]  # Interim analysis at 50% recruitment
effectiveness_threshold = 0.1

# Calculate adaptive EVSI
evsi_result = adaptive_evsi(
    sample_size=sample_size,
    interim_analysis_times=interim_analysis_times,
    effectiveness_threshold=effectiveness_threshold
)

print(f"Adaptive EVSI: {evsi_result:.2f}")
```

## Calibration VOI

Calibration VOI methods assess the value of information for model calibration studies.

### Usage Example

```python
import numpy as np
from voiage.methods.calibration import calibration_evppi

# Define calibration parameters
calibration_parameters = {
    "param1": np.random.normal(0, 1, 1000),
    "param2": np.random.normal(0, 1, 1000)
}

# Calculate calibration EVPPI
evppi_result = calibration_evppi(calibration_parameters)

print(f"Calibration EVPPI: {evppi_result:.2f}")
```

## Observational Study VOI

Observational study VOI methods evaluate the value of information from observational data.

### Usage Example

```python
import numpy as np
from voiage.methods.observational import observational_evsi

# Define observational study parameters
sample_size = 5000
confounding_variables = ["age", "gender", "comorbidities"]

# Calculate observational EVSI
evsi_result = observational_evsi(
    sample_size=sample_size,
    confounding_variables=confounding_variables
)

print(f"Observational EVSI: {evsi_result:.2f}")
```

## Sequential Decision VOI

Sequential decision VOI methods evaluate value of information in multi-stage decision problems.

### Usage Example

```python
import numpy as np
from voiage.methods.sequential import sequential_evpi

# Define sequential decision parameters
decision_stages = 3
transition_probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])

# Calculate sequential EVPI
evpi_result = sequential_evpi(
    decision_stages=decision_stages,
    transition_probabilities=transition_probabilities
)

print(f"Sequential EVPI: {evpi_result:.2f}")
```

## Structural Uncertainty VOI

Structural uncertainty VOI methods assess the value of information about model structure.

### Usage Example

```python
import numpy as np
from voiage.methods.structural import structural_evpi

# Define model structures and probabilities
model_structures = ["Model A", "Model B", "Model C"]
structure_probabilities = [0.4, 0.35, 0.25]

# Calculate structural EVPI
evpi_result = structural_evpi(
    model_structures=model_structures,
    structure_probabilities=structure_probabilities
)

print(f"Structural EVPI: {evpi_result:.2f}")
```

## Best Practices

1. **Method Selection**: Choose the appropriate VOI method for your specific research question
2. **Parameter Specification**: Carefully specify all relevant parameters for each method
3. **Validation**: Validate results with domain experts and sensitivity analysis
4. **Interpretation**: Interpret results in the context of decision-making