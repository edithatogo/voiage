"""Top-level public API for `voiage`.

The package exposes the curated core analysis surface together with the main
subpackage namespaces for advanced workflows.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from . import (
    analysis,
    backends,
    config,
    core,
    exceptions,
    factory,
    fluent,
    hta_integration,
    methods,
    schema,
)
from .analysis import DecisionAnalysis
from .methods.adaptive_learning_bandit import (
    AdaptiveLearningBanditResult,
    value_of_adaptive_learning_bandit,
)
from .methods.ai_assisted_evidence_triage import (
    AIAssistedEvidenceTriageResult,
    value_of_ai_assisted_evidence_triage,
)
from .methods.ambiguity_distribution_shift import (
    AmbiguityDistributionShiftResult,
    value_of_ambiguity_distribution_shift,
)
from .methods.basic import evpi, evppi
from .methods.capacity_budget_constrained import (
    CapacityBudgetConstrainedResult,
    value_of_capacity_budget_constrained,
)
from .methods.causal_transportability import (
    CausalTransportabilityResult,
    value_of_causal_transportability,
)
from .methods.ceaf import CEAFResult
from .methods.ceaf import calculate_ceaf as ceaf
from .methods.computational import (
    ComputationalResult,
    value_of_computational_refinement,
)
from .methods.data_quality import DataQualityResult, value_of_data_quality
from .methods.distributional import (
    DistributionalEquityResult,
    value_of_distributional_equity,
)
from .methods.dominance import DominanceResult
from .methods.dominance import calculate_dominance as dominance
from .methods.dynamic_real_options import (
    DynamicRealOptionsResult,
    value_of_dynamic_real_options,
)
from .methods.equity_information import (
    EquityInformationResult,
    value_of_equity_information,
)
from .methods.evidence_obsolescence_refresh import (
    EvidenceObsolescenceRefreshResult,
    value_of_evidence_obsolescence_refresh,
)
from .methods.expert_synthesis import ExpertSynthesisResult, value_of_expert_synthesis
from .methods.explainability_transparency import (
    ExplainabilityTransparencyResult,
    value_of_explainability_transparency,
)
from .methods.federated_privacy_preserving import (
    FederatedPrivacyPreservingResult,
    value_of_federated_privacy_preserving,
)
from .methods.implementation import (
    ImplementationAdjustedResult,
    value_of_implementation,
)
from .methods.implementation_strategy import (
    ImplementationStrategyComparisonResult,
    value_of_implementation_strategy_comparison,
)
from .methods.interoperability_standardization import (
    InteroperabilityStandardizationResult,
    value_of_interoperability_standardization,
)
from .methods.monitoring_surveillance import (
    MonitoringSurveillanceResult,
    value_of_monitoring_surveillance,
)
from .methods.perspective import (
    Perspective,
    PerspectiveSet,
    ValueOfPerspectiveResult,
    perspective_arrow_schema_fingerprint,
    perspective_result_to_arrow,
    value_of_perspective,
    write_perspective_result_ipc,
    write_perspective_result_parquet,
)
from .methods.preference import (
    PreferenceHeterogeneityResult,
    PreferenceProfile,
    PreferenceProfileSet,
    preference_optimal_strategies,
    value_of_preference,
    value_of_preference_heterogeneity,
    value_of_preference_information,
)
from .methods.regulatory_market_access import (
    RegulatoryMarketAccessResult,
    value_of_regulatory_market_access,
)
from .methods.replication_reproducibility import (
    ReplicationReproducibilityResult,
    value_of_replication_reproducibility,
)
from .methods.sample_information import enbs, evsi
from .methods.strategic_behavior import (
    StrategicBehaviorResult,
    value_of_strategic_behavior,
)
from .methods.threshold import (
    ThresholdProfile,
    ThresholdProfileSet,
    ThresholdResult,
    value_of_threshold,
    value_of_threshold_information,
)
from .methods.validation import (
    ModelValidationResult,
    ValidationProfile,
    ValidationProfileSet,
    value_of_model_validation,
    value_of_validation,
)
from .schema import (
    DecisionOption,
    ParameterSet,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
    ValueArray,
)

try:
    __version__ = _package_version("voiage")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.0.0"


_JAX_MODULES = frozenset({"health_economics", "multi_domain"})
_LAZY_MODULES = frozenset({"cli", "plot"})
_ECOSYSTEM_EXPORTS = {
    "ecosystem_integration": None,
    "HeomlRunBundle": "HeomlRunBundle",
    "load_heoml_run_bundle": "load_heoml_run_bundle",
}


def __getattr__(name: str) -> object:
    """Load provisional feature modules only when their extras are installed."""
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _ECOSYSTEM_EXPORTS:
        try:
            module = import_module(".ecosystem_integration", __name__)
        except ModuleNotFoundError as error:
            if (error.name or "").partition(".")[0] != "defusedxml":
                raise
            from .exceptions import raise_optional_dependency_error

            raise_optional_dependency_error(
                f"{name} requires the ecosystem integration dependencies; "
                "install them with `pip install 'voiage[ecosystem]'`."
            )
        export_name = _ECOSYSTEM_EXPORTS[name]
        value = module if export_name is None else getattr(module, export_name)
        globals()[name] = value
        return value
    if name not in _JAX_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f".{name}", __name__)
    except ModuleNotFoundError as error:
        if not (error.name or "").startswith("jax") and not str(error).startswith(
            "JAX intentionally absent"
        ):
            raise
        from .exceptions import raise_optional_dependency_error

        raise_optional_dependency_error(
            f"{name} requires JAX; install it with `pip install 'voiage[jax]'`."
        )
    globals()[name] = module
    return module


__all__ = [
    "AIAssistedEvidenceTriageResult",
    "AdaptiveLearningBanditResult",
    "AmbiguityDistributionShiftResult",
    "CEAFResult",
    "CapacityBudgetConstrainedResult",
    "CausalTransportabilityResult",
    "ComputationalResult",
    "DataQualityResult",
    "DecisionAnalysis",
    "DecisionOption",
    "DistributionalEquityResult",
    "DominanceResult",
    "DynamicRealOptionsResult",
    "EquityInformationResult",
    "EvidenceObsolescenceRefreshResult",
    "ExpertSynthesisResult",
    "ExplainabilityTransparencyResult",
    "FederatedPrivacyPreservingResult",
    "HeomlRunBundle",
    "ImplementationAdjustedResult",
    "ImplementationStrategyComparisonResult",
    "InteroperabilityStandardizationResult",
    "ModelValidationResult",
    "MonitoringSurveillanceResult",
    "ParameterSet",
    "Perspective",
    "PerspectiveSet",
    "perspective_arrow_schema_fingerprint",
    "perspective_result_to_arrow",
    "PortfolioSpec",
    "PortfolioStudy",
    "PreferenceHeterogeneityResult",
    "PreferenceProfile",
    "PreferenceProfileSet",
    "RegulatoryMarketAccessResult",
    "ReplicationReproducibilityResult",
    "StrategicBehaviorResult",
    "ThresholdProfile",
    "ThresholdProfileSet",
    "ThresholdResult",
    "TrialDesign",
    "ValidationProfile",
    "ValidationProfileSet",
    "ValueArray",
    "ValueOfPerspectiveResult",
    "analysis",
    "backends",
    "ceaf",
    "cli",
    "config",
    "core",
    "dominance",
    "ecosystem_integration",
    "enbs",
    "evpi",
    "evppi",
    "evsi",
    "exceptions",
    "factory",
    "fluent",
    "health_economics",
    "hta_integration",
    "load_heoml_run_bundle",
    "methods",
    "multi_domain",
    "plot",
    "preference_optimal_strategies",
    "schema",
    "value_of_adaptive_learning_bandit",
    "value_of_ai_assisted_evidence_triage",
    "value_of_ambiguity_distribution_shift",
    "value_of_capacity_budget_constrained",
    "value_of_causal_transportability",
    "value_of_computational_refinement",
    "value_of_data_quality",
    "value_of_distributional_equity",
    "value_of_dynamic_real_options",
    "value_of_equity_information",
    "value_of_evidence_obsolescence_refresh",
    "value_of_expert_synthesis",
    "value_of_explainability_transparency",
    "value_of_federated_privacy_preserving",
    "value_of_implementation",
    "value_of_implementation_strategy_comparison",
    "value_of_interoperability_standardization",
    "value_of_model_validation",
    "value_of_monitoring_surveillance",
    "value_of_perspective",
    "write_perspective_result_ipc",
    "write_perspective_result_parquet",
    "value_of_preference",
    "value_of_preference_heterogeneity",
    "value_of_preference_information",
    "value_of_regulatory_market_access",
    "value_of_replication_reproducibility",
    "value_of_strategic_behavior",
    "value_of_threshold",
    "value_of_threshold_information",
    "value_of_validation",
]
