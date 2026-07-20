use voiage_domain::SampleVector;

use crate::NumericalInputError;

const EFFECT_ATOL: f64 = 1.0e-8;
const EFFECT_RTOL: f64 = 1.0e-5;

/// Stable classification assigned to one strategy by dominance analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DominanceStatus {
    /// The strategy remains on the cost-effectiveness frontier.
    Frontier,
    /// A cheaper and no-less-effective alternative exists.
    StronglyDominated,
    /// The strategy is removed by the increasing-ICER frontier rule.
    ExtendedDominated,
}

/// Binding-independent dominance and cost-effectiveness frontier result.
#[derive(Clone, Debug, PartialEq)]
pub struct DominanceKernelResult {
    /// Original strategy indices in increasing-effect frontier order.
    pub frontier_indices: Vec<usize>,
    /// Strongly dominated original strategy indices in ascending order.
    pub strongly_dominated_indices: Vec<usize>,
    /// Extended-dominated original strategy indices in ascending order.
    pub extended_dominated_indices: Vec<usize>,
    /// Classification for each strategy in original input order.
    pub status: Vec<DominanceStatus>,
    /// Incremental costs in frontier transition order.
    pub incremental_costs: Vec<f64>,
    /// Incremental effects in frontier transition order.
    pub incremental_effects: Vec<f64>,
    /// Incremental cost-effectiveness ratios in frontier transition order.
    pub icers: Vec<f64>,
}

/// Computes strong dominance, the cost-effectiveness frontier, and ICERs.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when vector lengths differ or fewer than two
/// strategies are supplied. Numeric finiteness is enforced by [`SampleVector`].
pub fn dominance(
    costs: &SampleVector,
    effects: &SampleVector,
) -> Result<DominanceKernelResult, NumericalInputError> {
    if costs.len() != effects.len() {
        return Err(NumericalInputError::dimension(
            "effects",
            costs.len(),
            effects.len(),
            "cost and effect strategy counts must match",
        ));
    }
    if costs.len() < 2 {
        return Err(NumericalInputError::invalid(
            "costs",
            "at least two strategies are required",
        ));
    }

    let cost_values = costs.as_slice();
    let effect_values = effects.as_slice();
    let strongly_dominated_indices = strong_dominance(cost_values, effect_values);
    let frontier_indices = frontier(cost_values, effect_values, &strongly_dominated_indices);
    let extended_dominated_indices = (0..costs.len())
        .filter(|index| {
            !strongly_dominated_indices.contains(index) && !frontier_indices.contains(index)
        })
        .collect::<Vec<_>>();
    let mut status = vec![DominanceStatus::Frontier; costs.len()];
    for &index in &strongly_dominated_indices {
        status[index] = DominanceStatus::StronglyDominated;
    }
    for &index in &extended_dominated_indices {
        status[index] = DominanceStatus::ExtendedDominated;
    }
    let (incremental_costs, incremental_effects, icers) =
        incremental_results(cost_values, effect_values, &frontier_indices);

    Ok(DominanceKernelResult {
        frontier_indices,
        strongly_dominated_indices,
        extended_dominated_indices,
        status,
        incremental_costs,
        incremental_effects,
        icers,
    })
}

fn strong_dominance(costs: &[f64], effects: &[f64]) -> Vec<usize> {
    (0..costs.len())
        .filter(|&candidate| {
            (0..costs.len()).any(|alternative| {
                alternative != candidate
                    && costs[alternative] <= costs[candidate]
                    && effects[alternative] >= effects[candidate]
                    && (costs[alternative] < costs[candidate]
                        || effects[alternative] > effects[candidate])
            })
        })
        .collect()
}

fn frontier(costs: &[f64], effects: &[f64], strongly_dominated: &[usize]) -> Vec<usize> {
    let mut candidates = (0..costs.len())
        .filter(|index| !strongly_dominated.contains(index))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        effects[*left]
            .total_cmp(&effects[*right])
            .then_with(|| costs[*left].total_cmp(&costs[*right]))
            .then_with(|| left.cmp(right))
    });

    let mut result = Vec::with_capacity(candidates.len());
    for index in candidates {
        if let Some(previous) = result.last_mut() {
            if effect_is_close(effects[index], effects[*previous]) {
                if costs[index] < costs[*previous] {
                    *previous = index;
                }
                continue;
            }
        }
        result.push(index);
    }

    loop {
        let removable = (1..result.len().saturating_sub(1)).find(|&position| {
            incremental_icer(costs, effects, result[position - 1], result[position])
                >= incremental_icer(costs, effects, result[position], result[position + 1])
        });
        if let Some(position) = removable {
            result.remove(position);
        } else {
            break;
        }
    }
    result
}

fn effect_is_close(current: f64, previous: f64) -> bool {
    (current - previous).abs() <= EFFECT_ATOL + EFFECT_RTOL * previous.abs()
}

fn incremental_icer(costs: &[f64], effects: &[f64], previous: usize, next: usize) -> f64 {
    let incremental_effect = effects[next] - effects[previous];
    if incremental_effect <= 0.0 {
        f64::INFINITY
    } else {
        (costs[next] - costs[previous]) / incremental_effect
    }
}

fn incremental_results(
    costs: &[f64],
    effects: &[f64],
    frontier: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut incremental_costs = Vec::with_capacity(frontier.len().saturating_sub(1));
    let mut incremental_effects = Vec::with_capacity(frontier.len().saturating_sub(1));
    let mut icers = Vec::with_capacity(frontier.len().saturating_sub(1));
    for pair in frontier.windows(2) {
        let incremental_cost = costs[pair[1]] - costs[pair[0]];
        let incremental_effect = effects[pair[1]] - effects[pair[0]];
        incremental_costs.push(incremental_cost);
        incremental_effects.push(incremental_effect);
        icers.push(incremental_cost / incremental_effect);
    }
    (incremental_costs, incremental_effects, icers)
}
