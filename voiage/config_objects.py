"""Configuration objects for complex parameter sets in Value of Information analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VOIAnalysisConfig:
    """Configuration for a Value of Information analysis."""

    # Basic analysis parameters
    population: Optional[float] = None
    time_horizon: Optional[float] = None
    discount_rate: Optional[float] = None

    # Computational parameters
    chunk_size: Optional[int] = None
    use_jit: bool = False
    backend: str = "numpy"
    enable_caching: bool = False

    # Streaming parameters
    streaming_window_size: Optional[int] = None

    # EVPPI parameters
    n_regression_samples: Optional[int] = None
    regression_model: Optional[Any] = None

    # EVSI parameters
    n_simulations: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "population": self.population,
            "time_horizon": self.time_horizon,
            "discount_rate": self.discount_rate,
            "chunk_size": self.chunk_size,
            "use_jit": self.use_jit,
            "backend": self.backend,
            "enable_caching": self.enable_caching,
            "streaming_window_size": self.streaming_window_size,
            "n_regression_samples": self.n_regression_samples,
            "regression_model": self.regression_model,
            "n_simulations": self.n_simulations
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming data analysis."""

    window_size: int = 1000
    update_frequency: int = 100
    buffer_size: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
        if self.update_frequency <= 0:
            raise ValueError("Update frequency must be positive")
        if self.buffer_size is not None and self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")


@dataclass
class MetamodelConfig:
    """Configuration for metamodeling approaches."""

    # Metamodel type
    method: str = "gam"  # 'gam', 'gp', 'bart', 'rf', 'nn'

    # General parameters
    n_samples: int = 10000
    n_folds: int = 5

    # GAM-specific parameters
    gam_splines: int = 10
    gam_degree: int = 3

    # GP-specific parameters
    gp_length_scale: float = 1.0
    gp_noise_level: float = 0.1

    # BART-specific parameters
    bart_trees: int = 50
    bart_burnin: int = 100

    # RF-specific parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None

    # NN-specific parameters
    nn_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    nn_learning_rate: float = 0.001
    nn_epochs: int = 1000
    nn_batch_size: int = 32

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_methods = ['gam', 'gp', 'bart', 'rf', 'nn']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""

    # Optimization algorithm
    algorithm: str = "grid"  # 'grid', 'random', 'bayesian'

    # General parameters
    n_iterations: int = 100
    n_initial_points: int = 10

    # Grid search parameters
    grid_resolution: int = 10

    # Random search parameters
    random_seed: Optional[int] = None

    # Bayesian optimization parameters
    acquisition_function: str = "ei"  # 'ei', 'ucb', 'poi'
    kappa: float = 2.576  # For UCB
    xi: float = 0.01  # For EI and POI

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_algorithms = ['grid', 'random', 'bayesian']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")

        valid_acq_funcs = ['ei', 'ucb', 'poi']
        if self.acquisition_function not in valid_acq_funcs:
            raise ValueError(f"Acquisition function must be one of {valid_acq_funcs}")


@dataclass
class HealthcareConfig:
    """Configuration for healthcare-specific analyses."""

    # QALY parameters
    qaly_discount_rate: float = 0.03
    cost_discount_rate: float = 0.03

    # Disease progression parameters
    cycle_length: float = 1.0  # Years
    max_cycles: int = 50

    # Markov model parameters
    markov_cohort_size: int = 10000
    markov_start_age: float = 25.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 <= self.qaly_discount_rate <= 1):
            raise ValueError("QALY discount rate must be between 0 and 1")
        if not (0 <= self.cost_discount_rate <= 1):
            raise ValueError("Cost discount rate must be between 0 and 1")
        if self.cycle_length <= 0:
            raise ValueError("Cycle length must be positive")
        if self.max_cycles <= 0:
            raise ValueError("Max cycles must be positive")
        if self.markov_cohort_size <= 0:
            raise ValueError("Markov cohort size must be positive")
        if self.markov_start_age < 0:
            raise ValueError("Markov start age must be non-negative")


@dataclass
class EnvironmentalConfig:
    """Configuration for environmental impact assessments."""

    # Carbon footprint parameters
    carbon_intensity: float = 0.5  # kg CO2/kWh
    energy_consumption: float = 10000  # kWh

    # Water usage parameters
    water_intensity: float = 0.1  # L/kWh
    water_cost: float = 0.002  # $/L

    # Biodiversity parameters
    biodiversity_impact_factor: float = 0.01  # Impact per kWh

    # Monetization parameters
    social_cost_of_carbon: float = 50  # $/ton CO2
    ecosystem_service_value: float = 100  # $/impact unit

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.carbon_intensity < 0:
            raise ValueError("Carbon intensity must be non-negative")
        if self.energy_consumption < 0:
            raise ValueError("Energy consumption must be non-negative")
        if self.water_intensity < 0:
            raise ValueError("Water intensity must be non-negative")
        if self.water_cost < 0:
            raise ValueError("Water cost must be non-negative")
        if self.biodiversity_impact_factor < 0:
            raise ValueError("Biodiversity impact factor must be non-negative")
        if self.social_cost_of_carbon < 0:
            raise ValueError("Social cost of carbon must be non-negative")
        if self.ecosystem_service_value < 0:
            raise ValueError("Ecosystem service value must be non-negative")


@dataclass
class FinancialConfig:
    """Configuration for financial risk analysis."""

    # Risk metrics parameters
    var_confidence_level: float = 0.95
    cvar_confidence_level: float = 0.95
    sharpe_ratio_risk_free_rate: float = 0.0001  # Daily

    # Monte Carlo parameters
    mc_n_simulations: int = 10000
    mc_time_horizon: int = 252  # Trading days

    # Stress testing parameters
    stress_test_scenarios: List[str] = field(default_factory=lambda: ["market_crash", "interest_rate_shock"])

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 < self.var_confidence_level < 1):
            raise ValueError("VaR confidence level must be between 0 and 1")
        if not (0 < self.cvar_confidence_level < 1):
            raise ValueError("CVaR confidence level must be between 0 and 1")
        if self.mc_n_simulations <= 0:
            raise ValueError("Monte Carlo simulations must be positive")
        if self.mc_time_horizon <= 0:
            raise ValueError("Monte Carlo time horizon must be positive")


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    # Parallel processing parameters
    n_workers: Optional[int] = None
    use_processes: bool = True
    max_workers: Optional[int] = None

    # Memory management
    memory_limit_mb: Optional[float] = None
    chunk_size: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError("Number of workers must be positive")
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("Maximum workers must be positive")
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")


# Factory functions for creating common configurations
def create_default_config() -> VOIAnalysisConfig:
    """Create a default VOI analysis configuration."""
    return VOIAnalysisConfig()


def create_healthcare_config() -> HealthcareConfig:
    """Create a healthcare analysis configuration."""
    return HealthcareConfig()


def create_environmental_config() -> EnvironmentalConfig:
    """Create an environmental analysis configuration."""
    return EnvironmentalConfig()


def create_financial_config() -> FinancialConfig:
    """Create a financial analysis configuration."""
    return FinancialConfig()


def create_parallel_config() -> ParallelConfig:
    """Create a parallel processing configuration."""
    return ParallelConfig()


def create_streaming_config() -> StreamingConfig:
    """Create a streaming analysis configuration."""
    return StreamingConfig()


def create_metamodel_config(method: str = "gam") -> MetamodelConfig:
    """Create a metamodel configuration."""
    return MetamodelConfig(method=method)


def create_optimization_config(algorithm: str = "grid") -> OptimizationConfig:
    """Create an optimization configuration."""
    return OptimizationConfig(algorithm=algorithm)


# Example usage:
#
# # Create a comprehensive configuration
# config = VOIAnalysisConfig(
#     population=100000,
#     time_horizon=10,
#     discount_rate=0.03,
#     chunk_size=1000,
#     use_jit=True,
#     backend="jax",
#     enable_caching=True,
#     streaming_window_size=5000,
#     n_regression_samples=5000
# )
#
# # Use configuration in analysis
# analysis = DecisionAnalysis(
#     nb_array=net_benefits,
#     parameter_samples=parameters,
#     backend=config.backend,
#     use_jit=config.use_jit,
#     streaming_window_size=config.streaming_window_size,
#     enable_caching=config.enable_caching
# )
#
# evpi_result = analysis.evpi(
#     population=config.population,
#     time_horizon=config.time_horizon,
#     discount_rate=config.discount_rate,
#     chunk_size=config.chunk_size
# )
