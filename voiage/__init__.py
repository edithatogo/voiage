"""Top-level public API for `voiage`.

The package exposes the curated core analysis surface together with the main
subpackage namespaces for advanced workflows.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import TYPE_CHECKING

from . import (
    analysis,
    config,
    core,
    exceptions,
    factory,
    fluent,
    hta_integration,
    schema,
)
from .analysis import DecisionAnalysis
from .methods.basic import ExpectedLossResult, evpi, evppi, expected_loss
from .methods.ceaf import CEAFResult
from .methods.ceaf import calculate_ceaf as ceaf
from .methods.dominance import DominanceResult
from .methods.dominance import calculate_dominance as dominance
from .methods.sample_information import enbs, evsi
from .schema import (
    DecisionOption,
    ParameterSet,
    PortfolioSpec,
    PortfolioStudy,
    TrialDesign,
    ValueArray,
)

if TYPE_CHECKING:
    from . import ecosystem_integration
    from .ecosystem_integration import HeomlRunBundle, load_heoml_run_bundle

try:
    __version__ = _package_version("voiage")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.0.0"


_JAX_MODULES = frozenset({"health_economics", "multi_domain"})
_LAZY_MODULES = frozenset({"backends", "cli", "experimental", "methods", "plot"})
_ECOSYSTEM_EXPORTS = {
    "ecosystem_integration": None,
    "HeomlRunBundle": "HeomlRunBundle",
    "load_heoml_run_bundle": "load_heoml_run_bundle",
}
_EXTENSION_EXPORTS = {
    "AdaptiveLearningBanditResult": (
        ".methods.adaptive_learning_bandit",
        "AdaptiveLearningBanditResult",
    ),
    "value_of_adaptive_learning_bandit": (
        ".methods.adaptive_learning_bandit",
        "value_of_adaptive_learning_bandit",
    ),
    "AIAssistedEvidenceTriageResult": (
        ".methods.ai_assisted_evidence_triage",
        "AIAssistedEvidenceTriageResult",
    ),
    "value_of_ai_assisted_evidence_triage": (
        ".methods.ai_assisted_evidence_triage",
        "value_of_ai_assisted_evidence_triage",
    ),
    "AmbiguityDistributionShiftResult": (
        ".methods.ambiguity_distribution_shift",
        "AmbiguityDistributionShiftResult",
    ),
    "value_of_ambiguity_distribution_shift": (
        ".methods.ambiguity_distribution_shift",
        "value_of_ambiguity_distribution_shift",
    ),
    "CapacityBudgetConstrainedResult": (
        ".methods.capacity_budget_constrained",
        "CapacityBudgetConstrainedResult",
    ),
    "value_of_capacity_budget_constrained": (
        ".methods.capacity_budget_constrained",
        "value_of_capacity_budget_constrained",
    ),
    "CausalTransportabilityResult": (
        ".methods.causal_transportability",
        "CausalTransportabilityResult",
    ),
    "value_of_causal_transportability": (
        ".methods.causal_transportability",
        "value_of_causal_transportability",
    ),
    "ComputationalResult": (".methods.computational", "ComputationalResult"),
    "value_of_computational_refinement": (
        ".methods.computational",
        "value_of_computational_refinement",
    ),
    "DataQualityResult": (".methods.data_quality", "DataQualityResult"),
    "value_of_data_quality": (".methods.data_quality", "value_of_data_quality"),
    "DistributionalEquityResult": (
        ".methods.distributional",
        "DistributionalEquityResult",
    ),
    "value_of_distributional_equity": (
        ".methods.distributional",
        "value_of_distributional_equity",
    ),
    "DynamicRealOptionsResult": (
        ".methods.dynamic_real_options",
        "DynamicRealOptionsResult",
    ),
    "value_of_dynamic_real_options": (
        ".methods.dynamic_real_options",
        "value_of_dynamic_real_options",
    ),
    "EquityInformationResult": (
        ".methods.equity_information",
        "EquityInformationResult",
    ),
    "value_of_equity_information": (
        ".methods.equity_information",
        "value_of_equity_information",
    ),
    "EvidenceObsolescenceRefreshResult": (
        ".methods.evidence_obsolescence_refresh",
        "EvidenceObsolescenceRefreshResult",
    ),
    "value_of_evidence_obsolescence_refresh": (
        ".methods.evidence_obsolescence_refresh",
        "value_of_evidence_obsolescence_refresh",
    ),
    "ExpertSynthesisResult": (".methods.expert_synthesis", "ExpertSynthesisResult"),
    "value_of_expert_synthesis": (
        ".methods.expert_synthesis",
        "value_of_expert_synthesis",
    ),
    "ExplainabilityTransparencyResult": (
        ".methods.explainability_transparency",
        "ExplainabilityTransparencyResult",
    ),
    "value_of_explainability_transparency": (
        ".methods.explainability_transparency",
        "value_of_explainability_transparency",
    ),
    "FederatedPrivacyPreservingResult": (
        ".methods.federated_privacy_preserving",
        "FederatedPrivacyPreservingResult",
    ),
    "value_of_federated_privacy_preserving": (
        ".methods.federated_privacy_preserving",
        "value_of_federated_privacy_preserving",
    ),
    "ImplementationAdjustedResult": (
        ".methods.implementation",
        "ImplementationAdjustedResult",
    ),
    "value_of_implementation": (".methods.implementation", "value_of_implementation"),
    "ImplementationStrategyComparisonResult": (
        ".methods.implementation_strategy",
        "ImplementationStrategyComparisonResult",
    ),
    "value_of_implementation_strategy_comparison": (
        ".methods.implementation_strategy",
        "value_of_implementation_strategy_comparison",
    ),
    "InteroperabilityStandardizationResult": (
        ".methods.interoperability_standardization",
        "InteroperabilityStandardizationResult",
    ),
    "value_of_interoperability_standardization": (
        ".methods.interoperability_standardization",
        "value_of_interoperability_standardization",
    ),
    "MonitoringSurveillanceResult": (
        ".methods.monitoring_surveillance",
        "MonitoringSurveillanceResult",
    ),
    "value_of_monitoring_surveillance": (
        ".methods.monitoring_surveillance",
        "value_of_monitoring_surveillance",
    ),
    "Perspective": (".methods.perspective", "Perspective"),
    "PerspectiveSet": (".methods.perspective", "PerspectiveSet"),
    "ValueOfPerspectiveResult": (".methods.perspective", "ValueOfPerspectiveResult"),
    "perspective_arrow_schema_fingerprint": (
        ".methods.perspective",
        "perspective_arrow_schema_fingerprint",
    ),
    "perspective_result_to_arrow": (
        ".methods.perspective",
        "perspective_result_to_arrow",
    ),
    "value_of_perspective": (".methods.perspective", "value_of_perspective"),
    "write_perspective_result_ipc": (
        ".methods.perspective",
        "write_perspective_result_ipc",
    ),
    "write_perspective_result_parquet": (
        ".methods.perspective",
        "write_perspective_result_parquet",
    ),
    "PreferenceHeterogeneityResult": (
        ".methods.preference",
        "PreferenceHeterogeneityResult",
    ),
    "PreferenceProfile": (".methods.preference", "PreferenceProfile"),
    "PreferenceProfileSet": (".methods.preference", "PreferenceProfileSet"),
    "preference_optimal_strategies": (
        ".methods.preference",
        "preference_optimal_strategies",
    ),
    "value_of_preference": (".methods.preference", "value_of_preference"),
    "value_of_preference_heterogeneity": (
        ".methods.preference",
        "value_of_preference_heterogeneity",
    ),
    "value_of_preference_information": (
        ".methods.preference",
        "value_of_preference_information",
    ),
    "RegulatoryMarketAccessResult": (
        ".methods.regulatory_market_access",
        "RegulatoryMarketAccessResult",
    ),
    "value_of_regulatory_market_access": (
        ".methods.regulatory_market_access",
        "value_of_regulatory_market_access",
    ),
    "ReplicationReproducibilityResult": (
        ".methods.replication_reproducibility",
        "ReplicationReproducibilityResult",
    ),
    "value_of_replication_reproducibility": (
        ".methods.replication_reproducibility",
        "value_of_replication_reproducibility",
    ),
    "StrategicBehaviorResult": (
        ".methods.strategic_behavior",
        "StrategicBehaviorResult",
    ),
    "value_of_strategic_behavior": (
        ".methods.strategic_behavior",
        "value_of_strategic_behavior",
    ),
    "ThresholdProfile": (".methods.threshold", "ThresholdProfile"),
    "ThresholdProfileSet": (".methods.threshold", "ThresholdProfileSet"),
    "ThresholdResult": (".methods.threshold", "ThresholdResult"),
    "value_of_threshold": (".methods.threshold", "value_of_threshold"),
    "value_of_threshold_information": (
        ".methods.threshold",
        "value_of_threshold_information",
    ),
    "ModelValidationResult": (".methods.validation", "ModelValidationResult"),
    "ValidationProfile": (".methods.validation", "ValidationProfile"),
    "ValidationProfileSet": (".methods.validation", "ValidationProfileSet"),
    "value_of_model_validation": (".methods.validation", "value_of_model_validation"),
    "value_of_validation": (".methods.validation", "value_of_validation"),
}


def __getattr__(name: str) -> object:
    """Load provisional feature modules only when their extras are installed."""
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _EXTENSION_EXPORTS:
        module_name, export_name = _EXTENSION_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, export_name)
        globals()[name] = value
        return value
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


__all__ = [  # noqa: RUF022 - stable symbols precede provisional namespaces
    "CEAFResult",
    "DecisionAnalysis",
    "DecisionOption",
    "DominanceResult",
    "ExpectedLossResult",
    "ParameterSet",
    "PortfolioSpec",
    "PortfolioStudy",
    "TrialDesign",
    "ValueArray",
    "ceaf",
    "dominance",
    "enbs",
    "expected_loss",
    "evpi",
    "evppi",
    "evsi",
    "HeomlRunBundle",
    "analysis",
    "backends",
    "cli",
    "config",
    "core",
    "ecosystem_integration",
    "exceptions",
    "experimental",
    "factory",
    "fluent",
    "health_economics",
    "hta_integration",
    "load_heoml_run_bundle",
    "methods",
    "multi_domain",
    "plot",
    "schema",
]
