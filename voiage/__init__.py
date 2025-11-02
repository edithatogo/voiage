"""voiage: A Python library for Value of Information analysis.

This package provides tools for conducting Value of Information (VOI) analyses,
including Expected Value of Perfect Information (EVPI), Expected Value of
Partial Perfect Information (EVPPI), Expected Value of Sample Information (EVSI),
and Expected Net Benefit of Sampling (ENBS).

Modules:
    analysis: Core decision analysis functionality
    methods: Implementation of various VOI methods
    schema: Data structures for VOI analysis
    plot: Visualization tools for VOI results
    widgets: Interactive widgets for Jupyter notebooks
    web: Web API interface
    cli: Command-line interface
"""

__version__ = "0.2.0"

# Import key classes and functions for easy access
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray, ParameterSet
from voiage.methods.basic import evpi, evppi
from voiage.methods.sample_information import evsi
from voiage.methods.portfolio import portfolio_voi
from voiage.methods.sequential import sequential_voi
from voiage.methods.structural import structural_evpi, structural_evppi
from voiage.methods.calibration import voi_calibration
from voiage.methods.adaptive import adaptive_evsi
from voiage.methods.network_nma import evsi_nma
from voiage.methods.observational import voi_observational

# Import plotting functions
# These are imported lazily to avoid circular dependencies
# from voiage.plot.ceac import plot_ceac
# from voiage.plot.voi_curves import plot_evpi_vs_wtp as plot_evpi_curve

# Import widgets
from voiage.widgets.voi_widgets import VOIAnalysisWidget

# Import CLI app
from voiage.cli import app as cli_app

# Import web app
from voiage.web.main import app as web_app

# Import factory functions
from voiage.factory import (
    create_analysis,
    create_standard_analysis,
    create_streaming_analysis,
    create_healthcare_analysis,
    create_environmental_analysis,
    create_financial_analysis,
    create_large_scale_analysis,
    create_metamodel_analysis
)

# Import configuration objects
from voiage.config_objects import (
    VOIAnalysisConfig,
    HealthcareConfig,
    EnvironmentalConfig,
    FinancialConfig,
    ParallelConfig,
    StreamingConfig,
    MetamodelConfig,
    OptimizationConfig
)

# Import backends
from voiage.backends import get_backend, set_backend

# Import metamodels
from voiage.metamodels import (
    RandomForestMetamodel,
    GAMMetamodel,
    BARTMetamodel,
    PyTorchNNMetamodel,
    FlaxMetamodel,
    TinyGPMetamodel,
    EnsembleMetamodel
)

# Import core utilities
from voiage.core.gpu_acceleration import get_gpu_backend, is_gpu_available
from voiage.core.memory_optimization import optimize_value_array, optimize_parameter_set
from voiage.core.utils import check_input_array

# For convenience, make commonly used functions available at package level
__all__ = [
    # Core analysis
    "DecisionAnalysis",
    "ValueArray",
    "ParameterSet",
    
    # Basic VOI methods
    "evpi",
    "evppi",
    "evsi",
    
    # Advanced VOI methods
    "portfolio_voi",
    "sequential_voi",
    "structural_evpi",
    "structural_evppi",
    "voi_calibration",
    "adaptive_evsi",
    "evsi_nma",
    "voi_observational",
    
    # Plotting
    "plot_ceac",
    "plot_evpi_curve",
    
    # Widgets
    "VOIAnalysisWidget",
    
    # CLI and web
    "cli_app",
    "web_app",
    
    # Factory functions
    "create_analysis",
    "create_standard_analysis",
    "create_streaming_analysis",
    "create_healthcare_analysis",
    "create_environmental_analysis",
    "create_financial_analysis",
    "create_large_scale_analysis",
    "create_metamodel_analysis",
    
    # Configuration
    "VOIAnalysisConfig",
    "HealthcareConfig",
    "EnvironmentalConfig",
    "FinancialConfig",
    "ParallelConfig",
    "StreamingConfig",
    "MetamodelConfig",
    "OptimizationConfig",
    
    # Backends
    "get_backend",
    "set_backend",
    
    # Metamodels
    "RandomForestMetamodel",
    "GAMMetamodel",
    "BARTMetamodel",
    "PyTorchNNMetamodel",
    "FlaxMetamodel",
    "TinyGPMetamodel",
    "EnsembleMetamodel",
    
    # Core utilities
    "get_gpu_backend",
    "is_gpu_available",
    "optimize_value_array",
    "optimize_parameter_set",
    "check_input_array",
]