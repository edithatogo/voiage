use std::collections::HashSet;

use crate::NumericalInputError;

/// Candidate study in a research portfolio.
#[derive(Clone, Debug, PartialEq)]
pub struct PortfolioStudySpec {
    /// Unique study identifier.
    pub study_id: String,
    /// Total study cost (`>= 0.0`).
    pub cost: f64,
    /// Standalone EVSI (`>= 0.0`).
    pub standalone_evsi: f64,
    /// Parameter-group identifier used to detect overlapping information.
    pub parameter_group: String,
}

/// Explicit pairwise EVSI interaction adjustment when two studies inform overlapping parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortfolioInteraction {
    /// Index of first study (`< study_b`).
    pub study_a: usize,
    /// Index of second study (`> study_a`).
    pub study_b: usize,
    /// Joint EVSI adjustment (typically negative for redundant/overlapping evidence).
    pub joint_evsi_adjustment: f64,
}

/// Optimal feasible portfolio returned by exact small-portfolio enumeration.
#[derive(Clone, Debug, PartialEq)]
pub struct PortfolioOptimizationResult {
    /// Selected study indices in ascending order.
    pub selected_indices: Vec<usize>,
    /// Total research cost of the selected portfolio.
    pub total_cost: f64,
    /// Joint EVSI of the selected portfolio.
    pub joint_evsi: f64,
    /// Expected net benefit of sampling (`joint_evsi - total_cost`).
    pub enbs: f64,
    /// Number of feasible subsets evaluated.
    pub feasible_subsets_evaluated: usize,
}

/// Sequential adaptive design evaluation with stopping regret.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialDesignSummary {
    /// Index of the optimal stopping stage.
    pub optimal_stage: usize,
    /// Net value at the optimal stopping stage.
    pub optimal_net_value: f64,
    /// Regret of stopping immediately at stage 0 vs. the optimal stage.
    pub immediate_stop_regret: f64,
}

/// Enumerates all subsets of a small research portfolio (`<= 16` studies) under budget and overlap guards.
///
/// Rejects incompatible additive EVSI summation when two studies share a `parameter_group`
/// without an explicit [`PortfolioInteraction`] adjustment.
///
/// # Errors
///
/// Returns [`NumericalInputError`] on empty/oversized portfolios, invalid costs/EVSIs,
/// negative budgets, or unguarded parameter-group overlaps.
pub fn enumerate_study_portfolio(
    studies: &[PortfolioStudySpec],
    interactions: &[PortfolioInteraction],
    budget: f64,
    require_explicit_overlap_guard: bool,
) -> Result<PortfolioOptimizationResult, NumericalInputError> {
    if studies.is_empty() || studies.len() > 16 {
        return Err(NumericalInputError::invalid(
            "studies",
            "small-portfolio enumeration requires 1 to 16 studies",
        ));
    }
    if !budget.is_finite() || budget < 0.0 {
        return Err(NumericalInputError::invalid(
            "budget",
            "portfolio budget must be finite and non-negative",
        ));
    }
    for study in studies {
        if study.study_id.trim().is_empty() || study.parameter_group.trim().is_empty() {
            return Err(NumericalInputError::invalid(
                "studies",
                "study_id and parameter_group must be non-empty",
            ));
        }
        if !study.cost.is_finite()
            || study.cost < 0.0
            || !study.standalone_evsi.is_finite()
            || study.standalone_evsi < 0.0
        {
            return Err(NumericalInputError::invalid(
                "studies",
                "study cost and standalone_evsi must be finite and non-negative",
            ));
        }
    }

    let n = studies.len();
    let mut covered_pairs = HashSet::new();
    for inter in interactions {
        if inter.study_a >= inter.study_b || inter.study_b >= n {
            return Err(NumericalInputError::invalid(
                "interactions",
                "interaction indices must satisfy study_a < study_b < studies.len()",
            ));
        }
        if !inter.joint_evsi_adjustment.is_finite() {
            return Err(NumericalInputError::invalid(
                "interactions",
                "joint_evsi_adjustment must be finite",
            ));
        }
        if !covered_pairs.insert((inter.study_a, inter.study_b)) {
            return Err(NumericalInputError::invalid(
                "interactions",
                "duplicate pairwise interaction entry",
            ));
        }
    }

    if require_explicit_overlap_guard {
        for i in 0..n {
            for j in (i + 1)..n {
                if studies[i].parameter_group == studies[j].parameter_group
                    && !covered_pairs.contains(&(i, j))
                {
                    return Err(NumericalInputError::invalid(
                        "studies",
                        "overlapping parameter_group requires explicit PortfolioInteraction guard",
                    ));
                }
            }
        }
    }

    let subset_count = 1_usize << n;
    let mut best: Option<PortfolioOptimizationResult> = None;
    let mut feasible_subsets_evaluated = 0_usize;

    for mask in 0..subset_count {
        let mut total_cost = 0.0;
        let mut joint_evsi = 0.0;
        let mut selected = Vec::new();
        for (idx, study) in studies.iter().enumerate() {
            if (mask & (1_usize << idx)) != 0 {
                total_cost += study.cost;
                joint_evsi += study.standalone_evsi;
                selected.push(idx);
            }
        }
        if total_cost > budget {
            continue;
        }
        for inter in interactions {
            let has_a = (mask & (1_usize << inter.study_a)) != 0;
            let has_b = (mask & (1_usize << inter.study_b)) != 0;
            if has_a && has_b {
                joint_evsi += inter.joint_evsi_adjustment;
            }
        }
        let joint_evsi = joint_evsi.max(0.0);
        let enbs = joint_evsi - total_cost;
        feasible_subsets_evaluated += 1;

        let replace = match &best {
            None => true,
            Some(prev) => {
                enbs > prev.enbs + 1e-12
                    || ((enbs - prev.enbs).abs() <= 1e-12
                        && selected.len() < prev.selected_indices.len())
            }
        };
        if replace {
            best = Some(PortfolioOptimizationResult {
                selected_indices: selected,
                total_cost,
                joint_evsi,
                enbs,
                feasible_subsets_evaluated,
            });
        }
    }

    let mut result = best.ok_or_else(|| {
        NumericalInputError::invalid("budget", "no feasible portfolio subset within budget")
    })?;
    result.feasible_subsets_evaluated = feasible_subsets_evaluated;
    Ok(result)
}

/// Computes equity-weighted population ENBS across subgroups (`V10`, `F07`).
///
/// # Errors
///
/// Returns [`NumericalInputError`] if slices differ in length, are empty, contain
/// negative/non-finite values, or equity weights sum to zero.
pub fn equity_weighted_study_enbs(
    subgroup_evsi: &[f64],
    equity_weights: &[f64],
    research_cost: f64,
) -> Result<f64, NumericalInputError> {
    if subgroup_evsi.is_empty() || subgroup_evsi.len() != equity_weights.len() {
        return Err(NumericalInputError::invalid(
            "subgroup_evsi",
            "subgroup_evsi and equity_weights must be non-empty and equal length",
        ));
    }
    if !research_cost.is_finite() || research_cost < 0.0 {
        return Err(NumericalInputError::invalid(
            "research_cost",
            "research_cost must be finite and non-negative",
        ));
    }
    let mut weight_sum = 0.0;
    let mut weighted_evsi = 0.0;
    for (&evsi, &w) in subgroup_evsi.iter().zip(equity_weights.iter()) {
        if !evsi.is_finite() || evsi < 0.0 || !w.is_finite() || w < 0.0 {
            return Err(NumericalInputError::invalid(
                "equity_weights",
                "subgroup EVSI and equity weights must be finite and non-negative",
            ));
        }
        weight_sum += w;
        weighted_evsi += w * evsi;
    }
    if weight_sum <= 0.0 {
        return Err(NumericalInputError::invalid(
            "equity_weights",
            "equity weights must sum to a positive value",
        ));
    }
    Ok((weighted_evsi / weight_sum) - research_cost)
}

/// Evaluates sequential adaptive stopping stages with immediate-stop regret oracle (`V10`).
///
/// # Errors
///
/// Returns [`NumericalInputError`] for empty/mismatched slices or non-finite values.
pub fn sequential_adaptive_stopping_regret(
    stage_evsi_remaining: &[f64],
    stage_cumulative_costs: &[f64],
) -> Result<SequentialDesignSummary, NumericalInputError> {
    if stage_evsi_remaining.is_empty() || stage_evsi_remaining.len() != stage_cumulative_costs.len()
    {
        return Err(NumericalInputError::invalid(
            "stage_evsi_remaining",
            "stage slices must be non-empty and equal length",
        ));
    }
    let mut best_stage = 0_usize;
    let mut best_val = f64::NEG_INFINITY;
    let mut stage0_val = 0.0;
    for (idx, (&evsi, &cost)) in stage_evsi_remaining
        .iter()
        .zip(stage_cumulative_costs.iter())
        .enumerate()
    {
        if !evsi.is_finite() || evsi < 0.0 || !cost.is_finite() || cost < 0.0 {
            return Err(NumericalInputError::invalid(
                "stage_evsi_remaining",
                "stage EVSI and cumulative costs must be finite and non-negative",
            ));
        }
        let net = evsi - cost;
        if idx == 0 {
            stage0_val = net;
        }
        if net > best_val {
            best_val = net;
            best_stage = idx;
        }
    }
    Ok(SequentialDesignSummary {
        optimal_stage: best_stage,
        optimal_net_value: best_val,
        immediate_stop_regret: (best_val - stage0_val).max(0.0),
    })
}
