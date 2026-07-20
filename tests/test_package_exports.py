"""Regression coverage for curated subpackage exports."""

from __future__ import annotations

from importlib.metadata import version as package_version

import voiage
from voiage import (
    AmbiguityDistributionShiftResult,
    DecisionAnalysis,
    DecisionOption,
    DistributionalEquityResult,
    EquityInformationResult,
    ImplementationAdjustedResult,
    ParameterSet,
    Perspective,
    PerspectiveSet,
    PortfolioSpec,
    PortfolioStudy,
    PreferenceHeterogeneityResult,
    PreferenceProfile,
    PreferenceProfileSet,
    TrialDesign,
    ValueArray,
    ValueOfPerspectiveResult,
)
from voiage import analysis as analysis_module
from voiage import backends as backends_module
from voiage import cli as cli_module
from voiage import config as config_module
from voiage import core as core_module
from voiage import (
    enbs as top_level_enbs,
)
from voiage import (
    evpi as top_level_evpi,
)
from voiage import (
    evppi as top_level_evppi,
)
from voiage import (
    evsi as top_level_evsi,
)
from voiage import exceptions as exceptions_module
from voiage import factory as factory_module
from voiage import fluent as fluent_module
from voiage import health_economics as health_economics_module
from voiage import hta_integration as hta_integration_module
from voiage import methods as methods_module
from voiage import multi_domain as multi_domain_module
from voiage import plot as plot_module
from voiage import (
    preference_optimal_strategies as top_level_preference_optimal_strategies,
)
from voiage import schema as schema_module
from voiage import (
    value_of_ambiguity_distribution_shift as top_level_value_of_ambiguity_distribution_shift,
)
from voiage import (
    value_of_distributional_equity as top_level_value_of_distributional_equity,
)
from voiage import (
    value_of_equity_information as top_level_value_of_equity_information,
)
from voiage import (
    value_of_perspective as top_level_value_of_perspective,
)
from voiage import value_of_preference as top_level_value_of_preference
from voiage import (
    value_of_preference_heterogeneity as top_level_value_of_preference_heterogeneity,
)
from voiage import (
    value_of_preference_information as top_level_value_of_preference_information,
)
from voiage.core import (
    calculate_net_benefit,
    check_input_array,
    read_parameter_set_csv,
    read_value_array_csv,
    write_parameter_set_csv,
    write_value_array_csv,
)
from voiage.core.io import (
    read_parameter_set_csv as read_parameter_set_csv_impl,
)
from voiage.core.io import (
    read_value_array_csv as read_value_array_csv_impl,
)
from voiage.core.io import (
    write_parameter_set_csv as write_parameter_set_csv_impl,
)
from voiage.core.io import (
    write_value_array_csv as write_value_array_csv_impl,
)
from voiage.core.utils import (
    calculate_net_benefit as calculate_net_benefit_impl,
)
from voiage.core.utils import (
    check_input_array as check_input_array_impl,
)
from voiage.methods import (
    AmbiguityDistributionShiftResult as MethodsAmbiguityDistributionShiftResult,
)
from voiage.methods import (
    CEAFResult,
    DominanceResult,
    HeterogeneityResult,
    adaptive_evsi,
    calculate_ceaf,
    calculate_dominance,
    calculate_extended_dominance,
    calculate_icers,
    calculate_strong_dominance,
    cost_effectiveness_frontier,
    enbs,
    evpi,
    evppi,
    evsi,
    evsi_nma,
    identify_optimal_subgroups,
    perspective_optimal_strategies,
    portfolio_voi,
    sequential_voi,
    structural_evpi,
    structural_evppi,
    value_of_distributional_equity,
    value_of_heterogeneity,
    value_of_implementation,
    value_of_perspective,
    value_of_preference,
    value_of_preference_heterogeneity,
    value_of_preference_information,
    voi_calibration,
    voi_observational,
)
from voiage.methods import (
    DistributionalEquityResult as MethodsDistributionalEquityResult,
)
from voiage.methods import EquityInformationResult as MethodsEquityInformationResult
from voiage.methods import (
    ImplementationAdjustedResult as MethodsImplementationAdjustedResult,
)
from voiage.methods import (
    Perspective as MethodsPerspective,
)
from voiage.methods import (
    PerspectiveSet as MethodsPerspectiveSet,
)
from voiage.methods import (
    PreferenceHeterogeneityResult as MethodsPreferenceHeterogeneityResult,
)
from voiage.methods import PreferenceProfile as MethodsPreferenceProfile
from voiage.methods import PreferenceProfileSet as MethodsPreferenceProfileSet
from voiage.methods import (
    ValueOfPerspectiveResult as MethodsValueOfPerspectiveResult,
)
from voiage.methods.adaptive import adaptive_evsi as adaptive_evsi_impl
from voiage.methods.ambiguity_distribution_shift import (
    AmbiguityDistributionShiftResult as AmbiguityDistributionShiftResult_impl,
)
from voiage.methods.ambiguity_distribution_shift import (
    value_of_ambiguity_distribution_shift as value_of_ambiguity_distribution_shift_impl,
)
from voiage.methods.basic import evpi as evpi_impl
from voiage.methods.basic import evppi as evppi_impl
from voiage.methods.calibration import voi_calibration as voi_calibration_impl
from voiage.methods.ceaf import CEAFResult as CEAFResult_impl
from voiage.methods.ceaf import calculate_ceaf as calculate_ceaf_impl
from voiage.methods.distributional import (
    DistributionalEquityResult as DistributionalEquityResult_impl,
)
from voiage.methods.distributional import (
    value_of_distributional_equity as value_of_distributional_equity_impl,
)
from voiage.methods.dominance import DominanceResult as DominanceResult_impl
from voiage.methods.dominance import calculate_dominance as calculate_dominance_impl
from voiage.methods.dominance import (
    calculate_extended_dominance as calculate_extended_dominance_impl,
)
from voiage.methods.dominance import calculate_icers as calculate_icers_impl
from voiage.methods.dominance import (
    calculate_strong_dominance as calculate_strong_dominance_impl,
)
from voiage.methods.dominance import (
    cost_effectiveness_frontier as cost_effectiveness_frontier_impl,
)
from voiage.methods.equity_information import (
    EquityInformationResult as EquityInformationResult_impl,
)
from voiage.methods.equity_information import (
    value_of_equity_information as value_of_equity_information_impl,
)
from voiage.methods.heterogeneity import HeterogeneityResult as HeterogeneityResult_impl
from voiage.methods.heterogeneity import (
    identify_optimal_subgroups as identify_optimal_subgroups_impl,
)
from voiage.methods.heterogeneity import (
    value_of_heterogeneity as value_of_heterogeneity_impl,
)
from voiage.methods.implementation import (
    ImplementationAdjustedResult as ImplementationAdjustedResult_impl,
)
from voiage.methods.implementation import (
    value_of_implementation as value_of_implementation_impl,
)
from voiage.methods.network_nma import evsi_nma as evsi_nma_impl
from voiage.methods.observational import voi_observational as voi_observational_impl
from voiage.methods.perspective import Perspective as Perspective_impl
from voiage.methods.perspective import PerspectiveSet as PerspectiveSet_impl
from voiage.methods.perspective import (
    ValueOfPerspectiveResult as ValueOfPerspectiveResult_impl,
)
from voiage.methods.perspective import (
    perspective_optimal_strategies as perspective_optimal_strategies_impl,
)
from voiage.methods.perspective import (
    value_of_perspective as value_of_perspective_impl,
)
from voiage.methods.portfolio import portfolio_voi as portfolio_voi_impl
from voiage.methods.preference import (
    PreferenceHeterogeneityResult as PreferenceHeterogeneityResult_impl,
)
from voiage.methods.preference import PreferenceProfile as PreferenceProfile_impl
from voiage.methods.preference import PreferenceProfileSet as PreferenceProfileSet_impl
from voiage.methods.preference import (
    preference_optimal_strategies as preference_optimal_strategies_impl,
)
from voiage.methods.preference import (
    value_of_preference as value_of_preference_impl,
)
from voiage.methods.preference import (
    value_of_preference_heterogeneity as value_of_preference_heterogeneity_impl,
)
from voiage.methods.preference import (
    value_of_preference_information as value_of_preference_information_impl,
)
from voiage.methods.sample_information import enbs as enbs_impl
from voiage.methods.sample_information import evsi as evsi_impl
from voiage.methods.sequential import sequential_voi as sequential_voi_impl
from voiage.methods.structural import structural_evpi as structural_evpi_impl
from voiage.methods.structural import structural_evppi as structural_evppi_impl
from voiage.plot import (
    plot_ceac,
    plot_ceaf,
    plot_cost_effectiveness_plane,
    plot_evpi_vs_wtp,
    plot_evppi_surface,
    plot_evsi_vs_sample_size,
    plot_perspective_regret,
    plot_voh_by_subgroup,
)
from voiage.plot.ceac import plot_ceac as plot_ceac_impl
from voiage.plot.ceaf import plot_ceaf as plot_ceaf_impl
from voiage.plot.dominance import (
    plot_cost_effectiveness_plane as plot_cost_effectiveness_plane_impl,
)
from voiage.plot.heterogeneity import plot_voh_by_subgroup as plot_voh_by_subgroup_impl
from voiage.plot.perspective import (
    plot_perspective_regret as plot_perspective_regret_impl,
)
from voiage.plot.voi_curves import (
    plot_evpi_vs_wtp as plot_evpi_vs_wtp_impl,
)
from voiage.plot.voi_curves import (
    plot_evppi_surface as plot_evppi_surface_impl,
)
from voiage.plot.voi_curves import (
    plot_evsi_vs_sample_size as plot_evsi_vs_sample_size_impl,
)


def test_core_package_exports_point_to_leaf_implementations() -> None:
    """Core package exports should remain stable curated aliases."""
    assert calculate_net_benefit is calculate_net_benefit_impl
    assert check_input_array is check_input_array_impl
    assert read_parameter_set_csv is read_parameter_set_csv_impl
    assert read_value_array_csv is read_value_array_csv_impl
    assert write_parameter_set_csv is write_parameter_set_csv_impl
    assert write_value_array_csv is write_value_array_csv_impl


def test_methods_package_exports_point_to_leaf_implementations() -> None:
    """Method package exports should remain stable curated aliases."""
    assert CEAFResult is CEAFResult_impl
    assert DominanceResult is DominanceResult_impl
    assert DistributionalEquityResult is DistributionalEquityResult_impl
    assert AmbiguityDistributionShiftResult is AmbiguityDistributionShiftResult_impl
    assert (
        MethodsAmbiguityDistributionShiftResult is AmbiguityDistributionShiftResult_impl
    )
    assert EquityInformationResult is EquityInformationResult_impl
    assert MethodsEquityInformationResult is EquityInformationResult_impl
    assert ImplementationAdjustedResult is ImplementationAdjustedResult_impl
    assert MethodsDistributionalEquityResult is DistributionalEquityResult_impl
    assert MethodsImplementationAdjustedResult is ImplementationAdjustedResult_impl
    assert HeterogeneityResult is HeterogeneityResult_impl
    assert MethodsPerspective is Perspective_impl
    assert MethodsPerspectiveSet is PerspectiveSet_impl
    assert MethodsPreferenceHeterogeneityResult is PreferenceHeterogeneityResult_impl
    assert MethodsPreferenceProfile is PreferenceProfile_impl
    assert MethodsPreferenceProfileSet is PreferenceProfileSet_impl
    assert MethodsValueOfPerspectiveResult is ValueOfPerspectiveResult_impl
    assert adaptive_evsi is adaptive_evsi_impl
    assert calculate_ceaf is calculate_ceaf_impl
    assert calculate_dominance is calculate_dominance_impl
    assert calculate_extended_dominance is calculate_extended_dominance_impl
    assert calculate_icers is calculate_icers_impl
    assert calculate_strong_dominance is calculate_strong_dominance_impl
    assert cost_effectiveness_frontier is cost_effectiveness_frontier_impl
    assert identify_optimal_subgroups is identify_optimal_subgroups_impl
    assert perspective_optimal_strategies is perspective_optimal_strategies_impl
    assert enbs is enbs_impl
    assert evpi is evpi_impl
    assert evppi is evppi_impl
    assert evsi is evsi_impl
    assert evsi_nma is evsi_nma_impl
    assert portfolio_voi is portfolio_voi_impl
    assert sequential_voi is sequential_voi_impl
    assert structural_evpi is structural_evpi_impl
    assert structural_evppi is structural_evppi_impl
    assert value_of_distributional_equity is value_of_distributional_equity_impl
    assert value_of_implementation is value_of_implementation_impl
    assert value_of_heterogeneity is value_of_heterogeneity_impl
    assert value_of_perspective is value_of_perspective_impl
    assert value_of_preference is value_of_preference_impl
    assert value_of_preference_heterogeneity is value_of_preference_heterogeneity_impl
    assert value_of_preference_information is value_of_preference_information_impl
    assert top_level_preference_optimal_strategies is preference_optimal_strategies_impl
    assert voi_calibration is voi_calibration_impl
    assert voi_observational is voi_observational_impl


def test_backends_package_exports_are_curated() -> None:
    """Backend package exports should remain stable curated symbols."""
    assert backends_module.__all__ == [
        "JAX_AVAILABLE",
        "AppleMetalBackend",
        "Backend",
        "GpuAcceleration",
        "JaxAdvancedRegression",
        "JaxBackend",
        "JaxPerformanceProfiler",
        "NumpyBackend",
        "benchmark_evpi",
        "benchmark_memory_throughput",
        "benchmark_mps_vs_cpu",
        "compile_phase_3_handoff_packet",
        "get_backend",
        "set_backend",
    ]


def test_methods_package_exports_are_curated() -> None:
    """Method package exports should remain stable curated symbols."""
    assert methods_module.__all__ == [
        "AIAssistedEvidenceTriageResult",
        "AdaptiveLearningBanditResult",
        "AmbiguityDistributionShiftResult",
        "CEAFResult",
        "CapacityBudgetConstrainedResult",
        "CausalTransportabilityResult",
        "ComputationalResult",
        "DataQualityResult",
        "DistributionalEquityResult",
        "DominanceResult",
        "DynamicRealOptionsResult",
        "EquityInformationResult",
        "EvidenceObsolescenceRefreshResult",
        "ExpertSynthesisResult",
        "ExplainabilityTransparencyResult",
        "FederatedPrivacyPreservingResult",
        "HeterogeneityResult",
        "ImplementationAdjustedResult",
        "ImplementationStrategyComparisonResult",
        "InteroperabilityStandardizationResult",
        "ModelValidationResult",
        "MonitoringSurveillanceResult",
        "Perspective",
        "PerspectiveSet",
        "PreferenceHeterogeneityResult",
        "PreferenceProfile",
        "PreferenceProfileSet",
        "RegulatoryMarketAccessResult",
        "ReplicationReproducibilityResult",
        "StrategicBehaviorResult",
        "ThresholdProfile",
        "ThresholdProfileSet",
        "ThresholdResult",
        "ValidationProfile",
        "ValidationProfileSet",
        "ValueOfPerspectiveResult",
        "adaptive_evsi",
        "calculate_ceaf",
        "calculate_dominance",
        "calculate_extended_dominance",
        "calculate_icers",
        "calculate_strong_dominance",
        "cost_effectiveness_frontier",
        "enbs",
        "evpi",
        "evppi",
        "evsi",
        "evsi_nma",
        "identify_optimal_subgroups",
        "perspective_optimal_strategies",
        "portfolio_voi",
        "preference_optimal_strategies",
        "sequential_voi",
        "structural_evpi",
        "structural_evppi",
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
        "value_of_heterogeneity",
        "value_of_implementation",
        "value_of_implementation_strategy_comparison",
        "value_of_interoperability_standardization",
        "value_of_model_validation",
        "value_of_monitoring_surveillance",
        "value_of_perspective",
        "value_of_preference",
        "value_of_preference_heterogeneity",
        "value_of_preference_information",
        "value_of_regulatory_market_access",
        "value_of_replication_reproducibility",
        "value_of_strategic_behavior",
        "value_of_threshold",
        "value_of_threshold_information",
        "value_of_validation",
        "voi_calibration",
        "voi_observational",
    ]


def test_plot_package_exports_point_to_leaf_implementations() -> None:
    """Plot package exports should remain stable curated aliases."""
    assert plot_ceac is plot_ceac_impl
    assert plot_ceaf is plot_ceaf_impl
    assert plot_cost_effectiveness_plane is plot_cost_effectiveness_plane_impl
    assert plot_perspective_regret is plot_perspective_regret_impl
    assert plot_voh_by_subgroup is plot_voh_by_subgroup_impl
    assert plot_evpi_vs_wtp is plot_evpi_vs_wtp_impl
    assert plot_evppi_surface is plot_evppi_surface_impl
    assert plot_evsi_vs_sample_size is plot_evsi_vs_sample_size_impl


def test_top_level_package_exports_modules() -> None:
    """Top-level package exports should remain stable curated API symbols."""
    assert voiage.__all__ == [
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


def test_top_level_package_exports_point_to_modules() -> None:
    """Top-level package exports should remain stable module aliases."""
    assert voiage.analysis is analysis_module
    assert voiage.backends is backends_module
    assert voiage.cli is cli_module
    assert voiage.config is config_module
    assert voiage.core is core_module
    assert voiage.exceptions is exceptions_module
    assert voiage.factory is factory_module
    assert voiage.fluent is fluent_module
    assert voiage.health_economics is health_economics_module
    assert voiage.hta_integration is hta_integration_module
    assert voiage.methods is methods_module
    assert voiage.multi_domain is multi_domain_module
    assert voiage.plot is plot_module
    assert voiage.schema is schema_module
    assert DecisionAnalysis.__name__ == "DecisionAnalysis"
    assert DecisionOption.__name__ == "DecisionOption"
    assert ParameterSet.__name__ == "ParameterSet"
    assert DistributionalEquityResult is DistributionalEquityResult_impl
    assert Perspective is Perspective_impl
    assert PerspectiveSet is PerspectiveSet_impl
    assert PreferenceHeterogeneityResult is PreferenceHeterogeneityResult_impl
    assert PreferenceProfile is PreferenceProfile_impl
    assert PreferenceProfileSet is PreferenceProfileSet_impl
    assert PortfolioSpec.__name__ == "PortfolioSpec"
    assert PortfolioStudy.__name__ == "PortfolioStudy"
    assert TrialDesign.__name__ == "TrialDesign"
    assert ValueArray.__name__ == "ValueArray"
    assert ValueOfPerspectiveResult is ValueOfPerspectiveResult_impl
    assert top_level_preference_optimal_strategies is preference_optimal_strategies_impl
    assert top_level_value_of_preference is value_of_preference_impl
    assert (
        top_level_value_of_preference_heterogeneity
        is value_of_preference_heterogeneity_impl
    )
    assert (
        top_level_value_of_preference_information
        is value_of_preference_information_impl
    )
    assert top_level_evpi is evpi_impl
    assert top_level_evppi is evppi_impl
    assert top_level_evsi is evsi_impl
    assert top_level_enbs is enbs_impl
    assert (
        top_level_value_of_distributional_equity is value_of_distributional_equity_impl
    )
    assert top_level_value_of_equity_information is value_of_equity_information_impl
    assert (
        top_level_value_of_ambiguity_distribution_shift
        is value_of_ambiguity_distribution_shift_impl
    )
    assert top_level_value_of_perspective is value_of_perspective_impl
    assert voiage.__version__ == package_version("voiage")
