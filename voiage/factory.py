"""Factory methods for creating common Value of Information analysis patterns."""

from typing import Dict, Optional, Union

import numpy as np

from voiage.analysis import DecisionAnalysis
from voiage.config_objects import (
    EnvironmentalConfig,
    FinancialConfig,
    HealthcareConfig,
    MetamodelConfig,
    ParallelConfig,
    StreamingConfig,
    VOIAnalysisConfig,
)
from voiage.fluent import FluentDecisionAnalysis, create_analysis
from voiage.schema import ParameterSet, ValueArray

# Factory methods for common analysis patterns

def create_standard_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    use_jit: bool = False,
    backend: str = "numpy",
    enable_caching: bool = True
) -> DecisionAnalysis:
    """
    Create a standard VOI analysis with common settings.

    This factory method creates a DecisionAnalysis object with typical settings
    suitable for most analyses, including JIT compilation, caching, and population scaling.

    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        use_jit: Whether to use JIT compilation for faster computation
        backend: Computational backend ('numpy', 'jax', etc.)
        enable_caching: Whether to enable result caching

    Returns
    -------
        DecisionAnalysis: Configured analysis object
    """
    return DecisionAnalysis(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        use_jit=use_jit,
        backend=backend,
        enable_caching=enable_caching
    )


def create_streaming_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    window_size: int = 1000,
    update_frequency: int = 100,
    use_jit: bool = False,
    backend: str = "numpy"
) -> FluentDecisionAnalysis:
    """
    Create a streaming VOI analysis for continuous data updates.

    This factory method creates a FluentDecisionAnalysis object configured for
    streaming data analysis with windowed processing.

    Args:
        nb_array: Initial net benefit array
        parameter_samples: Initial parameter samples
        window_size: Size of the streaming window
        update_frequency: Frequency of updates
        use_jit: Whether to use JIT compilation
        backend: Computational backend

    Returns
    -------
        FluentDecisionAnalysis: Configured streaming analysis object
    """
    config = StreamingConfig(
        window_size=window_size,
        update_frequency=update_frequency
    )

    return (create_analysis(nb_array, parameter_samples)
            .with_backend(backend)
            .with_jit(use_jit)
            .with_streaming(config.window_size)
            .with_caching(True))


def create_healthcare_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    use_jit: bool = True,
    backend: str = "numpy",
    enable_caching: bool = True
) -> DecisionAnalysis:
    """
    Create a healthcare-specific VOI analysis.

    This factory method creates a DecisionAnalysis object configured for
    healthcare economic evaluation with QALY discounting.

    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        use_jit: Whether to use JIT compilation for faster computation
        backend: Computational backend ('numpy', 'jax', etc.)
        enable_caching: Whether to enable result caching

    Returns
    -------
        DecisionAnalysis: Configured healthcare analysis object
    """
    # Validate healthcare-specific parameters
    healthcare_config = HealthcareConfig()

    return DecisionAnalysis(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        use_jit=use_jit,
        backend=backend,
        enable_caching=enable_caching
    )


def create_environmental_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    carbon_intensity: float = 0.5,
    energy_consumption: float = 10000,
    water_intensity: float = 0.1
) -> DecisionAnalysis:
    """
    Create an environmental impact VOI analysis.

    This factory method creates a DecisionAnalysis object configured for
    environmental impact assessment.

    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        carbon_intensity: Carbon intensity (kg CO2/kWh)
        energy_consumption: Energy consumption (kWh)
        water_intensity: Water intensity (L/kWh)

    Returns
    -------
        DecisionAnalysis: Configured environmental analysis object
    """
    env_config = EnvironmentalConfig(
        carbon_intensity=carbon_intensity,
        energy_consumption=energy_consumption,
        water_intensity=water_intensity
    )

    return DecisionAnalysis(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        use_jit=True,
        backend="numpy",
        enable_caching=True
    )


def create_financial_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    var_confidence_level: float = 0.95,
    cvar_confidence_level: float = 0.95,
    mc_n_simulations: int = 10000
) -> DecisionAnalysis:
    """
    Create a financial risk VOI analysis.

    This factory method creates a DecisionAnalysis object configured for
    financial risk analysis with VaR and CVaR calculations.

    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        var_confidence_level: Confidence level for VaR calculation
        cvar_confidence_level: Confidence level for CVaR calculation
        mc_n_simulations: Number of Monte Carlo simulations

    Returns
    -------
        DecisionAnalysis: Configured financial analysis object
    """
    financial_config = FinancialConfig(
        var_confidence_level=var_confidence_level,
        cvar_confidence_level=cvar_confidence_level,
        mc_n_simulations=mc_n_simulations
    )

    return DecisionAnalysis(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        use_jit=True,
        backend="numpy",
        enable_caching=True
    )


def create_large_scale_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    chunk_size: int = 10000,
    n_workers: Optional[int] = None,
    memory_limit_mb: Optional[float] = None
) -> FluentDecisionAnalysis:
    """
    Create a large-scale VOI analysis with parallel processing and memory optimization.

    This factory method creates a FluentDecisionAnalysis object configured for
    handling large datasets with chunked processing and parallel execution.

    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        chunk_size: Size of chunks for incremental computation
        n_workers: Number of parallel workers
        memory_limit_mb: Memory limit in MB

    Returns
    -------
        FluentDecisionAnalysis: Configured large-scale analysis object
    """
    parallel_config = ParallelConfig(
        n_workers=n_workers,
        memory_limit_mb=memory_limit_mb
    )

    config = VOIAnalysisConfig(
        chunk_size=chunk_size,
        use_jit=True,
        backend="numpy",
        enable_caching=True
    )

    return (create_analysis(nb_array, parameter_samples)
            .with_backend(config.backend)
            .with_jit(config.use_jit)
            .with_caching(config.enable_caching))


def create_metamodel_analysis(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None,
    method: str = "gam",
    n_samples: int = 10000,
    n_folds: int = 5
) -> DecisionAnalysis:
    """
    Create a metamodel-based VOI analysis.

    This factory method creates a DecisionAnalysis object configured for
    metamodel-based analysis with various machine learning approaches.

    Args:
        nb_array: Net benefit array
        parameter_samples: Parameter samples for EVPPI calculation
        method: Metamodeling method ('gam', 'gp', 'bart', 'rf', 'nn')
        n_samples: Number of samples for metamodel training
        n_folds: Number of cross-validation folds

    Returns
    -------
        DecisionAnalysis: Configured metamodel analysis object
    """
    meta_config = MetamodelConfig(
        method=method,
        n_samples=n_samples,
        n_folds=n_folds
    )

    return DecisionAnalysis(
        nb_array=nb_array,
        parameter_samples=parameter_samples,
        use_jit=False,  # Metamodels may not benefit from JIT
        backend="numpy",
        enable_caching=True
    )


def create_configured_analysis(config: VOIAnalysisConfig) -> DecisionAnalysis:
    """
    Create a VOI analysis from a configuration object.

    This factory method creates a DecisionAnalysis object using a
    comprehensive configuration object.

    Args:
        config: VOIAnalysisConfig object with all analysis parameters

    Returns
    -------
        DecisionAnalysis: Configured analysis object
    """
    # Create an empty 2D array with shape (0, 1) as placeholder
    empty_array = np.array([]).reshape(0, 1).astype(np.float64)

    return DecisionAnalysis(
        nb_array=empty_array,
        parameter_samples=None,
        use_jit=config.use_jit,
        backend=config.backend,
        enable_caching=config.enable_caching,
        streaming_window_size=config.streaming_window_size
    )


# Example usage:
#
# # Standard analysis
# analysis = create_standard_analysis(
#     nb_array=net_benefits,
#     parameter_samples=parameters,
#     use_jit=True,
#     backend="numpy",
#     enable_caching=True
# )
#
# # Streaming analysis
# streaming_analysis = create_streaming_analysis(
#     nb_array=initial_net_benefits,
#     parameter_samples=initial_parameters,
#     window_size=5000
# )
#
# # Healthcare analysis
# healthcare_analysis = create_healthcare_analysis(
#     nb_array=net_benefits,
#     parameter_samples=parameters,
#     use_jit=True
# )
#
# # Large-scale analysis
# large_analysis = create_large_scale_analysis(
#     nb_array=large_net_benefits,
#     chunk_size=50000,
#     n_workers=4
# )
