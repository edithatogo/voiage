# Advanced VOI Methods

This guide covers the advanced Value of Information methods implemented in voiage, including adaptive trials, calibration, and observational studies.

The frontier methods described below span implemented, experimental, and
planned surfaces. The current fixture-backed contracts are intended to
stabilize the comparison surface, but call signatures, result payloads, and
reporting metadata may still evolve while the remaining frontier families are
brought online.

## Value Of Perspective

Value of Perspective compares the same strategy set under multiple decision
perspectives and exposes regret, consensus, and robust strategy summaries.

### Usage Example

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

# Net benefit samples with shape (n_samples, n_strategies, n_perspectives)
net_benefits = np.array(
    [
        [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
    ]
)

value_array = ValueArray.from_numpy_perspectives(
    net_benefits,
    strategy_names=["A", "B", "C"],
    perspective_names=["payer", "societal"],
)

analysis = DecisionAnalysis(value_array)
result = analysis.value_of_perspective()

print(result.consensus_strategy_name)
print(result.robust_strategy_name)
print(result.switching_values.tolist())
```

## Preference Heterogeneity And Individualized Care

Preference heterogeneity and individualized-care analysis are available through
the runtime surface, CLI entrypoint, and fixture-backed conformance contract.
Use them when decision value changes because patient, stakeholder, or subgroup
preferences matter directly.

### Usage Example

```python
from voiage.analysis import DecisionAnalysis
from voiage.methods.preference import PreferenceProfile, PreferenceProfileSet

result = DecisionAnalysis(...).value_of_preference(
    PreferenceProfileSet([
        PreferenceProfile("payer"),
        PreferenceProfile("patient"),
    ])
)

print(result.optimal_strategy_by_preference_profile["payer"])
```

CLI equivalent:

```bash
voiage calculate-preference preference_surface.json
```

## Model Validation And Threshold VOI

Validation VOI and threshold/robust VOI now have implemented runtime methods,
fixture-backed conformance payloads, and CLI entrypoints. The APIs are still
frontier work, so the comparison surface may evolve, but the current shape is
stable enough for side-by-side analysis and command-line smoke tests.

### Model Validation Example

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.methods.validation import ValidationProfile, ValidationProfileSet
from voiage.schema import ValueArray

net_benefits = np.array(
    [
        [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
    ]
)

value_array = ValueArray.from_numpy_perspectives(
    net_benefits,
    strategy_names=["A", "B", "C"],
    perspective_names=["external_validation", "discrepancy_reduction"],
)

analysis = DecisionAnalysis(value_array)
result = analysis.value_of_model_validation(
    validation_profiles=ValidationProfileSet(
        [
            ValidationProfile(
                id="external_validation",
                label="External validation",
                weight=0.6,
            ),
            ValidationProfile(
                id="discrepancy_reduction",
                label="Discrepancy reduction",
                weight=0.4,
            ),
        ]
    ),
    reference_validation_profile="external_validation",
)

print(result.consensus_strategy)
print(result.discrepancy_reduction_value)
```

### Threshold Example

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.methods.threshold import ThresholdProfile, ThresholdProfileSet
from voiage.schema import ValueArray

net_benefits = np.array(
    [
        [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
    ]
)

value_array = ValueArray.from_numpy_perspectives(
    net_benefits,
    strategy_names=["A", "B", "C"],
    perspective_names=["wtp_reversal", "policy_constraint"],
)

analysis = DecisionAnalysis(value_array)
result = analysis.value_of_threshold_information(
    threshold_profiles=ThresholdProfileSet(
        [
            ThresholdProfile(id="wtp_reversal", label="WTP reversal", weight=0.5),
            ThresholdProfile(
                id="policy_constraint",
                label="Policy constraint",
                weight=0.5,
            ),
        ]
    ),
    reference_threshold_profile="wtp_reversal",
)

print(result.tipping_point_strategy)
print(result.robust_strategy)
```

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

## Distributional And Equity-Weighted VOI

Distributional and equity-weighted VOI keeps subgroup-specific expected net
benefits explicit while summarizing equity-weighted welfare across subgroups.

### Usage Example

```python
import numpy as np
from voiage.methods.distributional import value_of_distributional_equity
from voiage.schema import ValueArray

values = np.array(
    [
        [10.0, 2.0],
        [8.0, 4.0],
        [1.0, 11.0],
        [2.0, 9.0],
    ]
)

value_array = ValueArray.from_numpy(values, ["A", "B"])
result = value_of_distributional_equity(
    value_array,
    subgroups=["low", "low", "high", "high"],
    equity_weights={"low": 0.25, "high": 0.75},
)

print(result.social_welfare_optimal_strategy_name)
print(result.social_welfare_value)
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
from voiage.methods.sequential import sequential_voi
from voiage.schema import DynamicSpec, ParameterSet

# Define sequential decision parameters
initial_psa = ParameterSet.from_numpy_or_dict(
    {
        "net_benefit_standard": np.array([0.0, 10.0, 0.0, 10.0]),
        "net_benefit_new": np.array([5.0, 5.0, 5.0, 5.0]),
    }
)
dynamic_spec = DynamicSpec(time_steps=[0.0, 1.0, 2.0])

# Calculate sequential VOI
sequential_result = sequential_voi(
    step_model=lambda psa, action, spec: {"next_psa": psa},
    initial_psa=initial_psa,
    dynamic_specification=dynamic_spec,
    wtp=50000.0,
)

print(f"Sequential VOI: {sequential_result:.2f}")
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
