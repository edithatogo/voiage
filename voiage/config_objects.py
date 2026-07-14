"""Configuration objects for complex parameter sets in Value of Information analysis."""

from dataclasses import dataclass, field
from typing import Any

from voiage.exceptions import raise_value_error

_WINDOW_SIZE_POSITIVE = "Window size must be positive"
_UPDATE_FREQUENCY_POSITIVE = "Update frequency must be positive"
_BUFFER_SIZE_POSITIVE = "Buffer size must be positive"
_QALY_DISCOUNT_RATE_RANGE = "QALY discount rate must be between 0 and 1"
_COST_DISCOUNT_RATE_RANGE = "Cost discount rate must be between 0 and 1"
_CYCLE_LENGTH_POSITIVE = "Cycle length must be positive"
_MAX_CYCLES_POSITIVE = "Max cycles must be positive"
_MARKOV_COHORT_SIZE_POSITIVE = "Markov cohort size must be positive"
_MARKOV_START_AGE_NON_NEGATIVE = "Markov start age must be non-negative"
_CARBON_INTENSITY_NON_NEGATIVE = "Carbon intensity must be non-negative"
_ENERGY_CONSUMPTION_NON_NEGATIVE = "Energy consumption must be non-negative"
_WATER_INTENSITY_NON_NEGATIVE = "Water intensity must be non-negative"
_WATER_COST_NON_NEGATIVE = "Water cost must be non-negative"
_BIODIVERSITY_IMPACT_FACTOR_NON_NEGATIVE = (
    "Biodiversity impact factor must be non-negative"
)
_SOCIAL_COST_OF_CARBON_NON_NEGATIVE = "Social cost of carbon must be non-negative"
_ECOSYSTEM_SERVICE_VALUE_NON_NEGATIVE = "Ecosystem service value must be non-negative"
_VAR_CONFIDENCE_LEVEL_RANGE = "VaR confidence level must be between 0 and 1"
_CVAR_CONFIDENCE_LEVEL_RANGE = "CVaR confidence level must be between 0 and 1"
_MC_SIMULATIONS_POSITIVE = "Monte Carlo simulations must be positive"
_MC_TIME_HORIZON_POSITIVE = "Monte Carlo time horizon must be positive"
_N_WORKERS_POSITIVE = "Number of workers must be positive"
_MAX_WORKERS_POSITIVE = "Maximum workers must be positive"
_MEMORY_LIMIT_POSITIVE = "Memory limit must be positive"
_CHUNK_SIZE_POSITIVE = "Chunk size must be positive"


def _choice_error(label: str, options: list[str]) -> str:
    """Format a stable validation error for enum-like configuration fields."""
    return f"{label} must be one of {options}"


@dataclass
class VOIAnalysisConfig:
    """Configuration for a Value of Information analysis."""

    # Basic analysis parameters
    population: float | None = None
    time_horizon: float | None = None
    discount_rate: float | None = None

    # Computational parameters
    chunk_size: int | None = None
    use_jit: bool = False
    backend: str = "numpy"
    enable_caching: bool = False

    # Streaming parameters
    streaming_window_size: int | None = None

    # EVPPI parameters
    n_regression_samples: int | None = None
    regression_model: Any | None = None

    # EVSI parameters
    n_simulations: int = 1000

    def to_dict(self) -> dict[str, Any]:
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
            "n_simulations": self.n_simulations,
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming data analysis."""

    window_size: int = 1000
    update_frequency: int = 100
    buffer_size: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_size <= 0:
            raise_value_error(_WINDOW_SIZE_POSITIVE)
        if self.update_frequency <= 0:
            raise_value_error(_UPDATE_FREQUENCY_POSITIVE)
        if self.buffer_size is not None and self.buffer_size <= 0:
            raise_value_error(_BUFFER_SIZE_POSITIVE)


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
    rf_max_depth: int | None = None

    # NN-specific parameters
    nn_hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    nn_learning_rate: float = 0.001
    nn_epochs: int = 1000
    nn_batch_size: int = 32

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_methods = ["gam", "gp", "bart", "rf", "nn"]
        if self.method not in valid_methods:
            raise_value_error(_choice_error("Method", valid_methods))


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
    random_seed: int | None = None

    # Bayesian optimization parameters
    acquisition_function: str = "ei"  # 'ei', 'ucb', 'poi'
    kappa: float = 2.576  # For UCB
    xi: float = 0.01  # For EI and POI

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        valid_algorithms = ["grid", "random", "bayesian"]
        if self.algorithm not in valid_algorithms:
            raise_value_error(_choice_error("Algorithm", valid_algorithms))

        valid_acq_funcs = ["ei", "ucb", "poi"]
        if self.acquisition_function not in valid_acq_funcs:
            raise_value_error(_choice_error("Acquisition function", valid_acq_funcs))


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

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not (0 <= self.qaly_discount_rate <= 1):
            raise_value_error(_QALY_DISCOUNT_RATE_RANGE)
        if not (0 <= self.cost_discount_rate <= 1):
            raise_value_error(_COST_DISCOUNT_RATE_RANGE)
        if self.cycle_length <= 0:
            raise_value_error(_CYCLE_LENGTH_POSITIVE)
        if self.max_cycles <= 0:
            raise_value_error(_MAX_CYCLES_POSITIVE)
        if self.markov_cohort_size <= 0:
            raise_value_error(_MARKOV_COHORT_SIZE_POSITIVE)
        if self.markov_start_age < 0:
            raise_value_error(_MARKOV_START_AGE_NON_NEGATIVE)


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

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.carbon_intensity < 0:
            raise_value_error(_CARBON_INTENSITY_NON_NEGATIVE)
        if self.energy_consumption < 0:
            raise_value_error(_ENERGY_CONSUMPTION_NON_NEGATIVE)
        if self.water_intensity < 0:
            raise_value_error(_WATER_INTENSITY_NON_NEGATIVE)
        if self.water_cost < 0:
            raise_value_error(_WATER_COST_NON_NEGATIVE)
        if self.biodiversity_impact_factor < 0:
            raise_value_error(_BIODIVERSITY_IMPACT_FACTOR_NON_NEGATIVE)
        if self.social_cost_of_carbon < 0:
            raise_value_error(_SOCIAL_COST_OF_CARBON_NON_NEGATIVE)
        if self.ecosystem_service_value < 0:
            raise_value_error(_ECOSYSTEM_SERVICE_VALUE_NON_NEGATIVE)


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
    stress_test_scenarios: list[str] = field(
        default_factory=lambda: ["market_crash", "interest_rate_shock"]
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not (0 < self.var_confidence_level < 1):
            raise_value_error(_VAR_CONFIDENCE_LEVEL_RANGE)
        if not (0 < self.cvar_confidence_level < 1):
            raise_value_error(_CVAR_CONFIDENCE_LEVEL_RANGE)
        if self.mc_n_simulations <= 0:
            raise_value_error(_MC_SIMULATIONS_POSITIVE)
        if self.mc_time_horizon <= 0:
            raise_value_error(_MC_TIME_HORIZON_POSITIVE)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    # Parallel processing parameters
    n_workers: int | None = None
    use_processes: bool = True
    max_workers: int | None = None

    # Memory management
    memory_limit_mb: float | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_workers is not None and self.n_workers <= 0:
            raise_value_error(_N_WORKERS_POSITIVE)
        if self.max_workers is not None and self.max_workers <= 0:
            raise_value_error(_MAX_WORKERS_POSITIVE)
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise_value_error(_MEMORY_LIMIT_POSITIVE)
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise_value_error(_CHUNK_SIZE_POSITIVE)


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
