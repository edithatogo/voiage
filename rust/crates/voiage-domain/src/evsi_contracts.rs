use serde::{Deserialize, Serialize};

use crate::CoreContractError;

/// Explicit sample-size convention for study-design EVSI calculations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "convention", rename_all = "snake_case")]
pub enum SampleSizeConvention {
    /// Sample size specified per arm in an equal-allocation two-arm study.
    PerArm {
        /// Number of participants per arm (`> 0`).
        per_arm: u32,
    },
    /// Total sample size across an equal-allocation two-arm study (`> 0` and even).
    TotalEqualTwoArm {
        /// Total participants across both arms.
        total: u32,
    },
    /// Explicit per-arm sample sizes for multi-arm studies (`>= 2` arms, all `> 0`).
    MultiArm {
        /// Participants allocated to each arm.
        per_arm_sizes: Vec<u32>,
    },
}

impl SampleSizeConvention {
    /// Validates the sample-size specification and rejects zero or odd-two-arm sizes.
    ///
    /// # Errors
    ///
    /// Returns [`CoreContractError`] when any arm size is zero, total equal two-arm
    /// size is odd, or a multi-arm specification has fewer than two arms.
    pub fn validate(&self) -> Result<(), CoreContractError> {
        match self {
            Self::PerArm { per_arm } => {
                if *per_arm == 0 {
                    return Err(CoreContractError::ZeroSampleSize);
                }
            }
            Self::TotalEqualTwoArm { total } => {
                if *total == 0 {
                    return Err(CoreContractError::ZeroSampleSize);
                }
                if total % 2 != 0 {
                    return Err(CoreContractError::Invalid("total_equal_two_arm_odd"));
                }
            }
            Self::MultiArm { per_arm_sizes } => {
                if per_arm_sizes.len() < 2 {
                    return Err(CoreContractError::Dimension {
                        field: "per_arm_sizes",
                        expected: 2,
                        actual: per_arm_sizes.len(),
                    });
                }
                if per_arm_sizes.contains(&0) {
                    return Err(CoreContractError::ZeroSampleSize);
                }
            }
        }
        Ok(())
    }

    /// Returns the total sample size across all arms.
    ///
    /// # Errors
    ///
    /// Returns [`CoreContractError`] when validation fails or total overflows `u32`.
    pub fn total_sample_size(&self) -> Result<u32, CoreContractError> {
        self.validate()?;
        match self {
            Self::PerArm { per_arm } => per_arm
                .checked_mul(2)
                .ok_or(CoreContractError::Invalid("total_sample_size_overflow")),
            Self::TotalEqualTwoArm { total } => Ok(*total),
            Self::MultiArm { per_arm_sizes } => per_arm_sizes
                .iter()
                .try_fold(0_u32, |acc, &n| acc.checked_add(n))
                .ok_or(CoreContractError::Invalid("total_sample_size_overflow")),
        }
    }

    /// Returns the per-arm sample sizes.
    ///
    /// # Errors
    ///
    /// Returns [`CoreContractError`] when validation fails.
    pub fn per_arm_sizes(&self) -> Result<Vec<u32>, CoreContractError> {
        self.validate()?;
        match self {
            Self::PerArm { per_arm } => Ok(vec![*per_arm, *per_arm]),
            Self::TotalEqualTwoArm { total } => {
                let half = *total / 2;
                Ok(vec![half, half])
            }
            Self::MultiArm { per_arm_sizes } => Ok(per_arm_sizes.clone()),
        }
    }
}

/// Supported EVSI estimator family identifiers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvsiEstimatorKind {
    /// Analytic conjugate study model.
    ConjugateAnalytic,
    /// Seeded bootstrap / nested Monte Carlo.
    NestedMonteCarlo,
    /// Regression-based EVSI metamodel.
    Regression,
    /// Moment-matching EVSI approximation.
    MomentBased,
    /// Importance-sampling / PSIS EVSI estimator.
    ImportanceSampling,
    /// Quasi-Monte Carlo (Halton) EVSI estimator.
    QmcHalton,
    /// Multilevel / paired-resolution EVSI estimator.
    Multilevel,
}

/// Coherent diagnostic summary attached to EVSI estimates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvsiDiagnostics {
    /// Effective sample size (`>= 0`, finite when present).
    pub ess: Option<f64>,
    /// Pareto-k tail diagnostic (`khat`, finite when present).
    pub khat: Option<f64>,
    /// Empirical decision-error rate in `[0, 1]` when evaluated against an oracle/baseline.
    pub decision_error: Option<f64>,
    /// Monte Carlo standard error (`>= 0`, finite when present).
    pub monte_carlo_se: Option<f64>,
    /// Estimated bias relative to reference (`finite` when present).
    pub bias_estimate: Option<f64>,
    /// Estimated root-mean-square error (`>= 0`, finite when present).
    pub rmse_estimate: Option<f64>,
    /// Empirical interval coverage in `[0, 1]` when intervals are supplied.
    pub coverage: Option<f64>,
}

impl EvsiDiagnostics {
    /// Validates finite and range constraints on all populated diagnostic fields.
    ///
    /// # Errors
    ///
    /// Returns [`CoreContractError`] if any field is non-finite or outside its domain.
    pub fn validate(&self) -> Result<(), CoreContractError> {
        for (name, val, min_zero) in [
            ("ess", self.ess, true),
            ("khat", self.khat, false),
            ("monte_carlo_se", self.monte_carlo_se, true),
            ("bias_estimate", self.bias_estimate, false),
            ("rmse_estimate", self.rmse_estimate, true),
        ] {
            if let Some(v) = val {
                if !v.is_finite() || (min_zero && v < 0.0) {
                    return Err(CoreContractError::Invalid(name));
                }
            }
        }
        for (name, prob) in [
            ("decision_error", self.decision_error),
            ("coverage", self.coverage),
        ] {
            if let Some(p) = prob {
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return Err(CoreContractError::Invalid(name));
                }
            }
        }
        Ok(())
    }
}
