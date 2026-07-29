//! Binding-independent numerical kernels for voiage.

#![forbid(unsafe_code)]

mod ceaf;
mod dominance;
mod enbs;
mod error;
mod evpi;
mod evppi;
mod evsi;
mod evsi_efficient;
mod evsi_moment;
mod evsi_regression;
mod expected_loss;
mod heterogeneity;
mod net_benefit;
mod replication;
mod structural;

pub use dominance::{dominance, DominanceKernelResult, DominanceStatus};
pub use enbs::enbs;
pub use error::NumericalInputError;
pub use evpi::{evpi, evpi_with_assurance, EvpiKernelResult};
pub use evppi::{evppi, evppi_with_assurance, EvppiKernelResult};
pub use evsi::{evsi_stochastic, EvsiKernelResult};
pub use evsi_efficient::{evsi_efficient_linear, EvsiApproximationResult};
pub use evsi_moment::evsi_moment_based;
pub use evsi_regression::{evsi_regression, EvsiRegressionResult};
pub use expected_loss::{expected_loss, ExpectedLossKernelResult};
pub use heterogeneity::{heterogeneity, HeterogeneityKernelResult};
pub use net_benefit::{net_benefit, NetBenefitKernelResult, WtpMode};
pub use replication::{summarize_replications, ReplicationSummary};
pub use structural::{
    structural_evpi, structural_evpi_with_assurance, structural_evppi,
    structural_evppi_with_assurance, StructuralVoiKernelResult,
};

/// Identifies this crate while numerical kernels are migrated.
pub const CRATE_NAME: &str = "voiage-numerics";
pub use ceaf::{ceaf, CeafKernelResult};
