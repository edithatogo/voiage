//! Experimental Q08 risk-prediction VOI and R01-R03 frontier research kernels.
//!
//! These kernels carry explicit experimental maturity bounds and are not
//! promoted to core public defaults without separate comparative qualification.

use crate::NumericalInputError;

/// Result of decision-aware simulation budget allocation (`R01`).
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAwareAllocationResult {
    /// Allocated simulation draws per candidate design/stratum.
    pub allocations: Vec<usize>,
    /// Estimated sequential error bound across strata.
    pub sequential_error_bound: f64,
    /// Total budget consumed.
    pub total_budget: usize,
}

/// Result of joint research-implementation-capacity optimization (`R02`).
#[derive(Clone, Debug, PartialEq)]
pub struct JointResearchImplementationResult {
    /// Selected research study index (`None` for no study).
    pub research_index: Option<usize>,
    /// Selected implementation-effort index.
    pub implementation_index: usize,
    /// Selected capacity-tier index.
    pub capacity_index: usize,
    /// Total net benefit after research, implementation, and capacity costs.
    pub net_benefit: f64,
}

/// Result of transport-bias-bounded EVSI sensitivity (`R03`, `V08`, `V11`).
#[derive(Clone, Debug, PartialEq)]
pub struct TransportBiasEvsiBound {
    /// Unadjusted target-population EVSI.
    pub nominal_evsi: f64,
    /// Worst-case lower bound under bounded transport shift `tv_bound`.
    pub lower_bound_evsi: f64,
    /// Worst-case upper bound under bounded transport shift `tv_bound`.
    pub upper_bound_evsi: f64,
    /// Experimental maturity tag.
    pub maturity: &'static str,
}

/// Computes decision-curve EVPI for a binary risk-prediction model (`Q08`, `V08`).
///
/// Given risk-threshold `threshold` in `(0, 1)` and uncertain true event-rate
/// draws `event_probabilities`, net benefit of treating all is `p - w * (1 - p)`
/// where `w = threshold / (1 - threshold)`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid thresholds, empty slices, or probabilities outside `[0, 1]`.
pub fn risk_prediction_threshold_evpi(
    event_probabilities: &[f64],
    threshold: f64,
) -> Result<f64, NumericalInputError> {
    if event_probabilities.is_empty() {
        return Err(NumericalInputError::invalid(
            "event_probabilities",
            "event_probabilities must be non-empty",
        ));
    }
    if !threshold.is_finite() || threshold <= 0.0 || threshold >= 1.0 {
        return Err(NumericalInputError::invalid(
            "threshold",
            "decision threshold must be strictly in (0, 1)",
        ));
    }
    let odds = threshold / (1.0 - threshold);
    let n = f64::from(u32::try_from(event_probabilities.len()).map_err(|_| {
        NumericalInputError::invalid("event_probabilities", "sample count exceeds u32")
    })?);
    let mut sum_treat_all = 0.0;
    let mut sum_perfect = 0.0;
    for &p in event_probabilities {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(NumericalInputError::invalid(
                "event_probabilities",
                "probabilities must be finite and in [0, 1]",
            ));
        }
        let nb_treat_all = p - odds * (1.0 - p);
        sum_treat_all += nb_treat_all / n;
        sum_perfect += nb_treat_all.max(0.0) / n;
    }
    let current = sum_treat_all.max(0.0);
    Ok((sum_perfect - current).max(0.0))
}

/// Allocates simulation budget across strata using decision-margin-aware weights (`R01`, `V09`, `F03`).
///
/// # Errors
///
/// Returns [`NumericalInputError`] for empty/mismatched slices or insufficient budget.
pub fn decision_aware_simulation_allocation(
    posterior_sds: &[f64],
    decision_margins: &[f64],
    total_budget: usize,
) -> Result<DecisionAwareAllocationResult, NumericalInputError> {
    let k = posterior_sds.len();
    if k == 0 || k != decision_margins.len() || total_budget < k {
        return Err(NumericalInputError::invalid(
            "total_budget",
            "strata must be non-empty, equal length, and total_budget >= strata count",
        ));
    }
    let mut scores = Vec::with_capacity(k);
    let mut score_sum = 0.0;
    for (&sd, &margin) in posterior_sds.iter().zip(decision_margins.iter()) {
        if !sd.is_finite() || sd < 0.0 || !margin.is_finite() {
            return Err(NumericalInputError::invalid(
                "posterior_sds",
                "posterior_sds and decision_margins must be finite",
            ));
        }
        let score = sd / (margin.abs() + 1e-6);
        scores.push(score);
        score_sum += score;
    }
    let mut allocations = vec![1_usize; k];
    let remaining = total_budget - k;
    let rem_u32 = u32::try_from(remaining)
        .map_err(|_| NumericalInputError::invalid("total_budget", "budget exceeds u32 range"))?;
    let mut assigned = 0_usize;
    for (idx, &score) in scores.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let extra = if score_sum > 0.0 {
            ((score / score_sum) * f64::from(rem_u32)).floor() as usize
        } else {
            remaining / k
        };
        allocations[idx] += extra;
        assigned += extra;
    }
    let mut leftover = remaining - assigned;
    let mut idx = 0_usize;
    while leftover > 0 {
        allocations[idx % k] += 1;
        leftover -= 1;
        idx += 1;
    }
    let mut max_se = 0.0_f64;
    for (&sd, &n) in posterior_sds.iter().zip(allocations.iter()) {
        let n_u32 = u32::try_from(n).map_err(|_| {
            NumericalInputError::invalid("total_budget", "allocation exceeds u32 range")
        })?;
        let se = sd / f64::from(n_u32).sqrt();
        max_se = max_se.max(se);
    }
    Ok(DecisionAwareAllocationResult {
        allocations,
        sequential_error_bound: max_se,
        total_budget,
    })
}

/// Exact small-case joint research, implementation, and capacity optimization (`R02`, `V06`, `V10`, `F06`).
///
/// `research_evsi_and_cost`: `[(evsi, cost)]`; `implementation_gain_and_cost`: `[(gain, cost, required_capacity)]`;
/// `capacity_levels_and_cost`: `[(capacity, cost)]`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] if implementation/capacity slices are empty, values are invalid,
/// or no combination satisfies the capacity constraint.
pub fn joint_research_implementation_capacity(
    research_evsi_and_cost: &[(f64, f64)],
    implementation_gain_and_cost: &[(f64, f64, f64)],
    capacity_levels_and_cost: &[(f64, f64)],
) -> Result<JointResearchImplementationResult, NumericalInputError> {
    if implementation_gain_and_cost.is_empty() || capacity_levels_and_cost.is_empty() {
        return Err(NumericalInputError::invalid(
            "implementation_gain_and_cost",
            "implementation and capacity options must be non-empty",
        ));
    }
    let mut best: Option<JointResearchImplementationResult> = None;

    let research_options: Vec<(Option<usize>, f64, f64)> = std::iter::once((None, 0.0, 0.0))
        .chain(
            research_evsi_and_cost
                .iter()
                .enumerate()
                .map(|(i, &(evsi, cost))| (Some(i), evsi, cost)),
        )
        .collect();

    for (r_idx, r_evsi, r_cost) in research_options {
        if !r_evsi.is_finite() || r_evsi < 0.0 || !r_cost.is_finite() || r_cost < 0.0 {
            return Err(NumericalInputError::invalid(
                "research_evsi_and_cost",
                "research EVSI and cost must be finite and non-negative",
            ));
        }
        for (i_idx, &(i_gain, i_cost, req_cap)) in implementation_gain_and_cost.iter().enumerate() {
            if !i_gain.is_finite()
                || !i_cost.is_finite()
                || i_cost < 0.0
                || !req_cap.is_finite()
                || req_cap < 0.0
            {
                return Err(NumericalInputError::invalid(
                    "implementation_gain_and_cost",
                    "implementation parameters must be finite and costs/capacity non-negative",
                ));
            }
            for (c_idx, &(cap, c_cost)) in capacity_levels_and_cost.iter().enumerate() {
                if !cap.is_finite() || cap < 0.0 || !c_cost.is_finite() || c_cost < 0.0 {
                    return Err(NumericalInputError::invalid(
                        "capacity_levels_and_cost",
                        "capacity and cost must be finite and non-negative",
                    ));
                }
                if cap + 1e-12 < req_cap {
                    continue;
                }
                let net = (r_evsi + i_gain) - (r_cost + i_cost + c_cost);
                if best.as_ref().is_none_or(|prev| net > prev.net_benefit) {
                    best = Some(JointResearchImplementationResult {
                        research_index: r_idx,
                        implementation_index: i_idx,
                        capacity_index: c_idx,
                        net_benefit: net,
                    });
                }
            }
        }
    }

    best.ok_or_else(|| {
        NumericalInputError::invalid(
            "capacity_levels_and_cost",
            "no capacity tier satisfies implementation requirements",
        )
    })
}

/// Bounds EVSI under bounded total-variation shift `tv_bound` in `[0, 1]` for a payoff
/// model with bounded span `payoff_span >= 0` (`R03`, `V08`, `V11`).
///
/// Because `EVSI = E[max_a E[U|Y]] - max_a E[U]` and each expectation shifts by at most
/// `tv_bound * payoff_span`, `|EVSI(Q) - EVSI(P)| <= 2 * tv_bound * payoff_span`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid EVSI, EVPI, `tv_bound`, or `payoff_span`.
pub fn transport_bias_evsi_bounds(
    nominal_evsi: f64,
    evpi_bound: f64,
    tv_bound: f64,
    payoff_span: f64,
) -> Result<TransportBiasEvsiBound, NumericalInputError> {
    if !nominal_evsi.is_finite()
        || nominal_evsi < 0.0
        || !evpi_bound.is_finite()
        || evpi_bound < nominal_evsi
    {
        return Err(NumericalInputError::invalid(
            "nominal_evsi",
            "require 0 <= nominal_evsi <= evpi_bound and both finite",
        ));
    }
    if !tv_bound.is_finite() || !(0.0..=1.0).contains(&tv_bound) {
        return Err(NumericalInputError::invalid(
            "tv_bound",
            "total-variation transport bound must be in [0, 1]",
        ));
    }
    if !payoff_span.is_finite() || payoff_span < 0.0 {
        return Err(NumericalInputError::invalid(
            "payoff_span",
            "payoff_span must be finite and non-negative",
        ));
    }
    let max_shift = 2.0 * tv_bound * payoff_span;
    let lower = (nominal_evsi - max_shift).max(0.0);
    let upper = (nominal_evsi + max_shift).min(evpi_bound);
    Ok(TransportBiasEvsiBound {
        nominal_evsi,
        lower_bound_evsi: lower,
        upper_bound_evsi: upper,
        maturity: "experimental_frontier_v1",
    })
}
