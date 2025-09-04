# Comprehensive API Reference

This document provides a complete reference for all public functions, classes, and modules in the voiage library.

## Table of Contents

1. [Core Modules](#core-modules)
   - [voiage.analysis](#voiageanalysis)
   - [voiage.schema](#voiageschema)
2. [Method Modules](#method-modules)
   - [voiage.methods.basic](#voiagemethodsbasic)
   - [voiage.methods.sample_information](#voiagemethodssample_information)
   - [voiage.methods.network_nma](#voiagemethodsnetwork_nma)
   - [voiage.methods.adaptive](#voiagemethodsadaptive)
   - [voiage.methods.calibration](#voiagemethodscalibration)
   - [voiage.methods.observational](#voiagemethodsobservational)
   - [voiage.methods.portfolio](#voiagemethodsportfolio)
   - [voiage.methods.sequential](#voiagemethodssequential)
   - [voiage.methods.structural](#voiagemethodsstructural)
3. [Utility Modules](#utility-modules)
   - [voiage.plot](#voiageplot)
   - [voiage.metamodels](#voiagemetamodels)
   - [voiage.exceptions](#voiageexceptions)

## Core Modules

### voiage.analysis

The core analysis module that provides the main DecisionAnalysis class for performing VOI calculations.

#### DecisionAnalysis

```python
class DecisionAnalysis:
    def __init__(self, nb_array: ValueArray, parameter_samples: ParameterSet):
        """
        Initialize a DecisionAnalysis object.
        
        Args:
            nb_array: ValueArray containing net benefits for each strategy and sample
            parameter_samples: ParameterSet containing parameter samples for PSA
        """
        pass
    
    def evpi(self) -> float:
        """
        Calculate Expected Value of Perfect Information (EVPI).
        
        Returns:
            float: EVPI value
        """
        pass
    
    def evppi(self, parameter_names: Optional[Union[str, List[str]]] = None) -> float:
        """
        Calculate Expected Value of Partial Perfect Information (EVPPI).
        
        Args:
            parameter_names: Name or list of parameter names for which to calculate EVPPI.
                           If None, calculates EVPPI for all parameters.
        
        Returns:
            float: EVPPI value
        """
        pass
    
    def evsi(self, parameter_names: Union[str, List[str]], 
             sample_size: int, n_inner_loops: int = 100) -> float:
        """
        Calculate Expected Value of Sample Information (EVSI).
        
        Args:
            parameter_names: Name or list of parameter names for which to calculate EVSI
            sample_size: Sample size for the proposed study
            n_inner_loops: Number of inner loops for Monte Carlo simulation
        
        Returns:
            float: EVSI value
        """
        pass
```

### voiage.schema

Data structures and schemas used throughout the library.

#### ValueArray

```python
class ValueArray:
    def __init__(self, dataset: xr.Dataset):
        """
        Initialize a ValueArray object.
        
        Args:
            dataset: xarray Dataset containing net benefit values
        """
        pass
    
    @classmethod
    def from_numpy(cls, values: np.ndarray, strategy_names: List[str]) -> 'ValueArray':
        """
        Create a ValueArray from numpy array.
        
        Args:
            values: 2D numpy array of shape (n_samples, n_strategies)
            strategy_names: List of strategy names
        
        Returns:
            ValueArray: New ValueArray object
        """
        pass
```

#### ParameterSet

```python
class ParameterSet:
    def __init__(self, parameters: Dict[str, np.ndarray]):
        """
        Initialize a ParameterSet object.
        
        Args:
            parameters: Dictionary mapping parameter names to numpy arrays of samples
        """
        pass
    
    @classmethod
    def from_numpy_or_dict(cls, data: Union[Dict[str, np.ndarray], np.ndarray]) -> 'ParameterSet':
        """
        Create a ParameterSet from numpy array or dictionary.
        
        Args:
            data: Dictionary of parameter arrays or numpy array
        
        Returns:
            ParameterSet: New ParameterSet object
        """
        pass
```

#### TrialDesign

```python
@dataclass
class TrialDesign:
    arms: List[DecisionOption]
```

#### DecisionOption

```python
@dataclass
class DecisionOption:
    name: str
    sample_size: int
```

## Method Modules

### voiage.methods.basic

Basic VOI calculation methods.

#### evpi

```python
def evpi(nb_array: ValueArray) -> float:
    """
    Calculate Expected Value of Perfect Information (EVPI).
    
    Args:
        nb_array: ValueArray containing net benefits for each strategy and sample
    
    Returns:
        float: EVPI value
    """
    pass
```

#### evppi

```python
def evppi(nb_array: ValueArray, parameter_samples: ParameterSet, 
          parameter_names: Optional[Union[str, List[str]]] = None) -> float:
    """
    Calculate Expected Value of Partial Perfect Information (EVPPI).
    
    Args:
        nb_array: ValueArray containing net benefits for each strategy and sample
        parameter_samples: ParameterSet containing parameter samples for PSA
        parameter_names: Name or list of parameter names for which to calculate EVPPI.
                        If None, calculates EVPPI for all parameters.
    
    Returns:
        float: EVPPI value
    """
    pass
```

### voiage.methods.sample_information

Sample information methods including EVSI calculation.

#### evsi_regression

```python
def evsi_regression(
    nb_array: ValueArray,
    parameter_samples: ParameterSet,
    parameter_names: Union[str, List[str]],
    sample_size: int,
    n_inner_loops: int = 100,
    **kwargs: Any
) -> float:
    """
    Calculate Expected Value of Sample Information (EVSI) using regression-based method.
    
    Args:
        nb_array: ValueArray containing net benefits for each strategy and sample
        parameter_samples: ParameterSet containing parameter samples for PSA
        parameter_names: Name or list of parameter names for which to calculate EVSI
        sample_size: Sample size for the proposed study
        n_inner_loops: Number of inner loops for Monte Carlo simulation
        **kwargs: Additional arguments
    
    Returns:
        float: EVSI value
    """
    pass
```

#### evsi_two_loop

```python
def evsi_two_loop(
    nb_array: ValueArray,
    parameter_samples: ParameterSet,
    parameter_names: Union[str, List[str]],
    sample_size: int,
    n_outer_loops: int = 20,
    n_inner_loops: int = 100,
    **kwargs: Any
) -> float:
    """
    Calculate Expected Value of Sample Information (EVSI) using two-loop Monte Carlo method.
    
    Args:
        nb_array: ValueArray containing net benefits for each strategy and sample
        parameter_samples: ParameterSet containing parameter samples for PSA
        parameter_names: Name or list of parameter names for which to calculate EVSI
        sample_size: Sample size for the proposed study
        n_outer_loops: Number of outer loops for Monte Carlo simulation
        n_inner_loops: Number of inner loops for Monte Carlo simulation
        **kwargs: Additional arguments
    
    Returns:
        float: EVSI value
    """
    pass
```

### voiage.methods.network_nma

Network Meta-Analysis VOI methods.

#### evsi_nma

```python
def evsi_nma(
    nma_model_evaluator: NMAEconomicModelEvaluator,
    psa_prior_nma: PSASample,
    trial_design_new_study: TrialDesign,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    n_inner_loops: int = 100,
    **kwargs: Any
) -> float:
    """
    Calculate the Expected Value of Sample Information for a new study in the context of a Network Meta-Analysis (EVSI-NMA).
    
    Args:
        nma_model_evaluator: A complex callable that encapsulates the NMA and subsequent economic evaluation
        psa_prior_nma: PSA samples representing current (prior) uncertainty about all relevant parameters
        trial_design_new_study: Specification of the new study whose data would inform the NMA
        population: Population size for scaling the EVSI to a population
        discount_rate: Discount rate for scaling (0-1)
        time_horizon: Time horizon for scaling in years
        n_outer_loops: Number of outer loops for Monte Carlo simulation
        n_inner_loops: Number of inner loops for Monte Carlo simulation
        **kwargs: Additional arguments for the NMA simulation or EVSI calculation method
    
    Returns:
        float: The calculated EVSI-NMA value
    """
    pass
```

### voiage.methods.adaptive

Adaptive trial VOI methods.

#### adaptive_evsi

```python
def adaptive_evsi(
    adaptive_trial_simulator: AdaptiveTrialEconomicSim,
    psa_prior: PSASample,
    base_trial_design: TrialDesign,
    adaptive_rules: Dict[str, Any],
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    n_inner_loops: int = 100,
    **kwargs: Any
) -> float:
    """
    Calculate the Expected Value of Sample Information for an adaptive trial design.
    
    Args:
        adaptive_trial_simulator: A function that simulates an adaptive trial and evaluates outcomes
        psa_prior: PSA samples representing current (prior) uncertainty
        base_trial_design: Base trial design before adaptations
        adaptive_rules: Rules for adaptive modifications
        population: Population size for scaling
        discount_rate: Discount rate for scaling
        time_horizon: Time horizon for scaling
        n_outer_loops: Number of outer loops for Monte Carlo simulation
        n_inner_loops: Number of inner loops for Monte Carlo simulation
        **kwargs: Additional arguments
    
    Returns:
        float: The calculated adaptive EVSI value
    """
    pass
```

### voiage.methods.calibration

Model calibration VOI methods.

#### voi_calibration

```python
def voi_calibration(
    cal_study_modeler: CalibrationStudyModeler,
    psa_prior: PSASample,
    calibration_study_design: Dict[str, Any],
    calibration_process_spec: Dict[str, Any],
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    **kwargs: Any
) -> float:
    """
    Calculate the Value of Information for data collected for Model Calibration.
    
    Args:
        cal_study_modeler: A function that simulates the calibration data collection
        psa_prior: PSA samples representing current (prior) uncertainty
        calibration_study_design: Specification of the data collection effort for calibration
        calibration_process_spec: Details of the calibration algorithm itself
        population: Population size for scaling
        discount_rate: Discount rate for scaling
        time_horizon: Time horizon for scaling
        n_outer_loops: Number of outer loops for Monte Carlo simulation
        **kwargs: Additional arguments
    
    Returns:
        float: The calculated VOI for the calibration study
    """
    pass
```

### voiage.methods.observational

Observational study VOI methods.

#### voi_observational

```python
def voi_observational(
    obs_study_modeler: ObservationalStudyModeler,
    psa_prior: PSASample,
    observational_study_design: Dict[str, Any],
    bias_models: Dict[str, Any],
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    **kwargs: Any
) -> float:
    """
    Calculate the Value of Information for collecting Observational Data (VOI-OS).
    
    Args:
        obs_study_modeler: A function that simulates the collection of observational data
        psa_prior: PSA samples representing current (prior) uncertainty
        observational_study_design: A detailed specification of the proposed observational study
        bias_models: Specifications for how biases will be modeled and quantitatively adjusted for
        population: Population size for scaling
        discount_rate: Discount rate for scaling
        time_horizon: Time horizon for scaling
        n_outer_loops: Number of outer loops for Monte Carlo simulation
        **kwargs: Additional arguments
    
    Returns:
        float: The calculated VOI for the observational study
    """
    pass
```

### voiage.methods.portfolio

Portfolio optimization methods.

#### portfolio_optimization

```python
def portfolio_optimization(
    returns: np.ndarray,
    constraints: Optional[Dict[str, Any]] = None,
    risk_tolerance: float = 1.0,
    method: str = "mean_variance"
) -> Dict[str, Any]:
    """
    Perform portfolio optimization.
    
    Args:
        returns: Array of returns for different assets
        constraints: Optimization constraints
        risk_tolerance: Risk tolerance parameter
        method: Optimization method to use
    
    Returns:
        Dict[str, Any]: Optimization results
    """
    pass
```

### voiage.methods.sequential

Sequential VOI methods.

#### sequential_voi

```python
def sequential_voi(
    decision_model: Callable,
    parameter_samples: ParameterSet,
    decision_horizon: int,
    discount_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate sequential VOI over a decision horizon.
    
    Args:
        decision_model: Function that evaluates decisions over time
        parameter_samples: ParameterSet containing parameter samples
        decision_horizon: Number of decision periods
        discount_rate: Discount rate for future values
    
    Returns:
        Dict[str, Any]: Sequential VOI results
    """
    pass
```

### voiage.methods.structural

Structural VOI methods.

#### structural_voi

```python
def structural_voi(
    model_family: List[Callable],
    parameter_samples: ParameterSet,
    model_weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate structural VOI for model uncertainty.
    
    Args:
        model_family: List of alternative model structures
        parameter_samples: ParameterSet containing parameter samples
        model_weights: Weights for different models
    
    Returns:
        float: Structural VOI value
    """
    pass
```

## Utility Modules

### voiage.plot

Plotting functions for VOI analysis.

#### ceac

```python
def ceac(nb_array: ValueArray, wtp_range: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot Cost-Effectiveness Acceptability Curve (CEAC).
    
    Args:
        nb_array: ValueArray containing net benefits
        wtp_range: Range of willingness-to-pay values
    
    Returns:
        plt.Figure: CEAC plot
    """
    pass
```

#### voi_curves

```python
def voi_curves(sample_sizes: np.ndarray, voi_values: np.ndarray) -> plt.Figure:
    """
    Plot VOI curves showing value as a function of sample size.
    
    Args:
        sample_sizes: Array of sample sizes
        voi_values: Array of corresponding VOI values
    
    Returns:
        plt.Figure: VOI curves plot
    """
    pass
```

### voiage.metamodels

Metamodeling functions for VOI analysis.

#### fit_metamodel

```python
def fit_metamodel(
    parameter_samples: ParameterSet,
    output_values: np.ndarray,
    method: str = "gam",
    **kwargs: Any
) -> Any:
    """
    Fit a metamodel to approximate complex simulation models.
    
    Args:
        parameter_samples: ParameterSet containing input parameter samples
        output_values: Array of output values from simulation model
        method: Metamodeling method to use
        **kwargs: Additional arguments for the metamodeling method
    
    Returns:
        Any: Fitted metamodel
    """
    pass
```

### voiage.exceptions

Custom exceptions for the library.

#### InputError

```python
class InputError(Exception):
    """Exception raised for invalid input parameters."""
    pass
```

#### ConvergenceError

```python
class ConvergenceError(Exception):
    """Exception raised when an algorithm fails to converge."""
    pass
```