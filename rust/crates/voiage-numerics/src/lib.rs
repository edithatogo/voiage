//! Binding-independent numerical kernels for voiage.

#![forbid(unsafe_code)]

mod ceaf;
mod credal;
mod dominance;
mod enbs;
mod error;
mod estimation_variance;
mod evpi;
mod evpi_resume;
mod evppi;
mod evsi;
mod evsi_efficient;
mod evsi_importance;
mod evsi_moment;
mod evsi_normal_normal;
mod evsi_qmc;
mod evsi_regression;
mod evsi_resume;
mod evsi_study_models;
mod frontier_research;
mod heterogeneity;
mod sequential_evsi;
mod signed_social_information;
mod simulation_outcome_adapter;
mod structural;
mod study_design;
mod study_portfolio;
mod tobit;
mod utility_information;

pub use credal::{
    solve_credal, verify_e_admissibility, verify_strict_credal_dominance, CredalAnalysis,
    CredalBenefitRange, CredalExtremum, CredalSolveResult,
};
pub use dominance::{dominance, DominanceKernelResult, DominanceStatus};
pub use enbs::{discounted_population_enbs, enbs, StudyScheduleConfig};
pub use error::NumericalInputError;
pub use estimation_variance::{
    estimation_truth_assurance, evppi_variance, evppi_variance_with_assurance, evsi_variance,
    evsi_variance_with_assurance, EstimationTruthAssuranceKernelResult,
    EstimationVarianceKernelResult,
};
pub use evpi::evpi;
pub use evpi_resume::{evpi_resumable, ResumableEvpiOutcome};
pub use evppi::{
    evppi, evppi_from_mars_artifact, evppi_nonlinear, EvppiBasis, EvppiMarsResult,
    EvppiNonlinearConfig, EvppiNonlinearResult, MarsArtifact, MarsHingeFactor, MarsStrategyModel,
    MarsTerm,
};
pub use evsi::{evsi_stochastic, EvsiKernelResult};
pub use evsi_efficient::{evsi_efficient_linear, EvsiApproximationResult};
pub use evsi_importance::{
    evsi_importance_sampling, evsi_importance_sampling_raw, EvsiImportanceResult,
};
pub use evsi_moment::evsi_moment_based;
pub use evsi_normal_normal::normal_normal_two_arm_evsi;
pub use evsi_qmc::{
    compare_evsi_mc_vs_qmc, evsi_multilevel_normal_normal, evsi_qmc_normal_normal,
    EvsiAccuracyCostComparison, EvsiQmcResult,
};
pub use evsi_regression::{evsi_regression, EvsiRegressionResult};
pub use evsi_resume::{evsi_streaming_resumable, ResumableEvsiOutcome};
pub use evsi_study_models::{
    beta_binomial_two_arm_evsi, exponential_time_to_event_evsi, gamma_poisson_evsi,
};
pub use frontier_research::{
    decision_aware_simulation_allocation, joint_research_implementation_capacity,
    risk_prediction_threshold_evpi, transport_bias_evsi_bounds, DecisionAwareAllocationResult,
    JointResearchImplementationResult, TransportBiasEvsiBound,
};
pub use heterogeneity::{heterogeneity, HeterogeneityKernelResult};
pub use sequential_evsi::{sequential_evsi, ObservationChannel, SequentialEvsi};
pub use signed_social_information::{
    evaluate_signed_social_information, SignedDesignEvaluation, SignedInformationDesign,
    SignedInformationInput, SignedInformationLedger, SignedInformationSelector, SignedWelfareStage,
};
pub use simulation_outcome_adapter::{
    adapt_simulation_outcome, adapt_simulation_outcome_input, simulation_outcome_evpi,
    SimulationOutcomeAdapterError, SimulationOutcomeEvpiResult, SimulationOutcomeHandoff,
};
pub use structural::{structural_evpi, structural_evppi};
pub use study_design::{
    coss, coss_selection_uncertainty, evsi_evpi_efficiency, information_efficiency_uncertainty,
    CossKernelResult, CossSelectionUncertaintyKernelResult, InformationEfficiencyKernelResult,
    InformationEfficiencyUncertaintyKernelResult,
};
pub use study_portfolio::{
    enumerate_study_portfolio, equity_weighted_study_enbs, sequential_adaptive_stopping_regret,
    PortfolioInteraction, PortfolioOptimizationResult, PortfolioStudySpec, SequentialDesignSummary,
};
pub use tobit::{TobitModel, TobitObservation, TobitState};
pub use utility_information::{
    expected_utility_information, AffineReduction, Comparability, DomainExclusion,
    ExpectedUtilityError, ExpectedUtilityInformationInput, ExpectedUtilityInformationResult,
    InformationStructure, MeasureResult, PolicyEvaluation, PolicyResult, PolicyTransition,
    RankingEquivalence, RootResult, SolverSettings, UtilityDescriptor,
};

/// Identifies this crate while numerical kernels are migrated.
pub const CRATE_NAME: &str = "voiage-numerics";
pub use ceaf::{ceaf, CeafKernelResult};
