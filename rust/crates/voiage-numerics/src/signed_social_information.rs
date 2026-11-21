//! Numerical evaluation of admitted finite signed-information policy tables.

use std::collections::BTreeSet;

use voiage_domain::{round_to_binary64, BigRational, SampleMatrix};

use crate::NumericalInputError;

/// The ledger used for the declared social objective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignedWelfareStage {
    /// Score the expected payoff before transfers and costs.
    PreTransfer,
    /// Score the payoff after transfers and costs.
    PostTransfer,
}

/// The score used when comparing policies within a design.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignedInformationSelector {
    /// Use the welfare weights and the declared ledger stage.
    SocialWelfare,
    /// Use the indexed agent's payoff after transfers and costs.
    Agent(usize),
}

/// One design's admitted policies and accounting inputs.
#[derive(Clone, Copy, Debug)]
pub struct SignedInformationDesign<'a> {
    /// Policy table indices in tie-breaking order. Exact ties use the first.
    pub policy_indices: &'a [usize],
    /// Use the given policy index when selection has already been declared.
    pub selected_policy: Option<usize>,
    /// Objective used to score all policies in the design.
    pub selector: SignedInformationSelector,
    /// Comparator design index. Use the design's own index for its baseline.
    pub comparator: usize,
    /// Transfers in declared order as (payer, recipient, amount).
    pub transfers: &'a [(usize, usize, f64)],
    /// Resource costs in declared order as (agent, amount).
    pub costs: &'a [(usize, f64)],
}

/// Finite numerical inputs in the caller's declared world and agent order.
///
/// Each policy matrix gives its permitted action's payoffs as [world][agent].
/// The caller admits the world law, observations, welfare units and rights
/// receipts before selecting those rows, and binds agent, policy and design
/// identifiers to the corresponding positions.
#[derive(Clone, Copy, Debug)]
pub struct SignedInformationInput<'a> {
    /// The complete probability law, in the payoff tables' world order.
    pub world_probabilities: &'a [f64],
    /// One finite payoff table for each admitted policy.
    pub policy_payoffs: &'a [SampleMatrix],
    /// Welfare weights in the payoff tables' agent order.
    pub welfare_weights: &'a [f64],
    /// Agent indices for each role, in the caller's role order.
    pub role_agents: &'a [Vec<usize>],
    /// Declared ledger used for welfare selection and signed differences.
    pub welfare_stage: SignedWelfareStage,
    /// Designs in comparator index order.
    pub designs: &'a [SignedInformationDesign<'a>],
}

/// The selected policy's ledger, in declared agent order.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedInformationLedger {
    /// Expected utility before transfers and costs.
    pub pre_transfer: Vec<f64>,
    /// Signed transfers. Payers have negative entries.
    pub transfer: Vec<f64>,
    /// Nonnegative resource costs.
    pub cost: Vec<f64>,
    /// Expected utility after adding transfers and subtracting costs.
    pub post_transfer: Vec<f64>,
}

/// Absolute and comparator-relative results for one design.
#[derive(Clone, Debug, PartialEq)]
pub struct SignedDesignEvaluation {
    /// Policy scores in the design's policy index order.
    pub selector_values: Vec<f64>,
    /// The selected policy's index in the full input policy table.
    pub selected_policy: usize,
    /// The selected policy's absolute account balances.
    pub ledger: SignedInformationLedger,
    /// Social welfare before transfers and costs.
    pub social_pre_transfer: f64,
    /// Social welfare after transfers and costs.
    pub social_post_transfer: f64,
    /// Signed post-transfer differences from the comparator, by agent.
    pub signed_by_agent: Vec<f64>,
    /// Signed post-transfer differences summed over each role's members.
    pub signed_by_role: Vec<f64>,
    /// Weighted signed differences at the declared welfare stage.
    pub signed_welfare: f64,
}

/// Evaluates the admitted finite policy catalog for every design.
///
/// Products and sequential ledger entries use binary64 arithmetic. Sums
/// accumulate those represented terms exactly and round once. World
/// probabilities are used as supplied after the v1 mass check.
///
/// # Errors
///
/// Returns an input error for an incomplete probability law, inconsistent
/// table shapes, invalid indices or accounting, or non-finite arithmetic.
pub fn evaluate_signed_social_information(
    input: &SignedInformationInput<'_>,
) -> Result<Vec<SignedDesignEvaluation>, NumericalInputError> {
    validate_input(input)?;
    let payoffs = input
        .policy_payoffs
        .iter()
        .map(|policy| {
            (0..input.welfare_weights.len())
                .map(|agent| {
                    finite_sum(
                        "policy_payoffs",
                        input
                            .world_probabilities
                            .iter()
                            .zip(policy.rows())
                            .map(|(probability, world)| probability * world[agent]),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut evaluations = input
        .designs
        .iter()
        .map(|design| evaluate_design(input, design, &payoffs))
        .collect::<Result<Vec<_>, _>>()?;

    // A comparator's policy may be evaluated after the design that uses it.
    for (index, design) in input.designs.iter().enumerate() {
        let ledger = &evaluations[index].ledger;
        let comparator = &evaluations[design.comparator].ledger;
        let by_agent = differences(&ledger.post_transfer, &comparator.post_transfer)?;
        let by_role = input
            .role_agents
            .iter()
            .map(|role| finite_sum("role_values", role.iter().map(|&agent| by_agent[agent])))
            .collect::<Result<_, _>>()?;
        let welfare = match input.welfare_stage {
            SignedWelfareStage::PreTransfer => weighted_sum(
                input.welfare_weights,
                &differences(&ledger.pre_transfer, &comparator.pre_transfer)?,
            )?,
            SignedWelfareStage::PostTransfer => weighted_sum(input.welfare_weights, &by_agent)?,
        };
        let result = &mut evaluations[index];
        result.signed_by_agent = by_agent;
        result.signed_by_role = by_role;
        result.signed_welfare = welfare;
    }
    Ok(evaluations)
}

fn validate_input(input: &SignedInformationInput<'_>) -> Result<(), NumericalInputError> {
    let probabilities = input.world_probabilities;
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|&p| !p.is_finite() || p <= 0.0 || p > 1.0)
    {
        return Err(NumericalInputError::invalid(
            "world_probabilities",
            "World probabilities must be positive and at most one",
        ));
    }
    let mass = finite_sum("world_probabilities", probabilities.iter().copied())?;
    if (mass - 1.0).abs() > 1.0e-9 * mass.max(1.0) {
        return Err(NumericalInputError::invalid(
            "world_probabilities",
            "World probabilities must sum to one",
        ));
    }
    let weights = input.welfare_weights;
    if weights.iter().any(|&w| !w.is_finite() || w < 0.0) || !weights.iter().any(|&w| w > 0.0) {
        return Err(NumericalInputError::invalid(
            "welfare_weights",
            "Welfare weights must be finite and nonnegative with a positive weight",
        ));
    }
    if input.policy_payoffs.is_empty() || input.designs.is_empty() {
        return Err(NumericalInputError::invalid(
            "designs",
            "Signed information requires policies and designs",
        ));
    }
    for payoffs in input.policy_payoffs {
        let [worlds, agents] = payoffs.shape();
        if worlds != probabilities.len() {
            return Err(NumericalInputError::dimension(
                "policy_payoffs",
                probabilities.len(),
                worlds,
                "Policy payoff rows must match the world law",
            ));
        }
        if agents != weights.len() {
            return Err(NumericalInputError::dimension(
                "policy_payoffs",
                weights.len(),
                agents,
                "Policy payoff columns must match the agents",
            ));
        }
    }
    for role in input.role_agents {
        check_indices("role_agents", role, weights.len())?;
    }
    for design in input.designs {
        if design.comparator >= input.designs.len() {
            return Err(NumericalInputError::invalid(
                "comparator",
                "Comparator must reference an evaluated design",
            ));
        }
        if design.policy_indices.is_empty() {
            return Err(NumericalInputError::invalid(
                "policy_indices",
                "A design catalog must contain at least one policy",
            ));
        }
        check_indices(
            "policy_indices",
            design.policy_indices,
            input.policy_payoffs.len(),
        )?;
        if let Some(policy) = design.selected_policy {
            if !design.policy_indices.contains(&policy) {
                return Err(NumericalInputError::invalid(
                    "selected_policy",
                    "The selected policy must belong to the design catalog",
                ));
            }
        }
        if let SignedInformationSelector::Agent(agent) = design.selector {
            if agent >= weights.len() {
                return Err(NumericalInputError::invalid(
                    "selector",
                    "The selector must reference a declared agent",
                ));
            }
        }
    }
    Ok(())
}

fn check_indices(
    field: &'static str,
    indices: &[usize],
    count: usize,
) -> Result<(), NumericalInputError> {
    let mut seen = BTreeSet::new();
    if indices
        .iter()
        .any(|&index| index >= count || !seen.insert(index))
    {
        return Err(NumericalInputError::invalid(
            field,
            "Indices must be distinct and in range",
        ));
    }
    Ok(())
}

fn evaluate_design(
    input: &SignedInformationInput<'_>,
    design: &SignedInformationDesign<'_>,
    payoffs: &[Vec<f64>],
) -> Result<SignedDesignEvaluation, NumericalInputError> {
    let (transfer, cost) = design_accounts(input.welfare_weights.len(), design)?;
    let mut scores = Vec::with_capacity(design.policy_indices.len());
    let mut winner = 0;
    let mut winning_post = Vec::new();
    for (position, &policy) in design.policy_indices.iter().enumerate() {
        let pre = &payoffs[policy];
        let post = pre
            .iter()
            .zip(&transfer)
            .zip(&cost)
            .map(|((&payoff, &transfer), &cost)| finite("post_transfer", payoff + transfer - cost))
            .collect::<Result<Vec<_>, _>>()?;
        let score = match design.selector {
            SignedInformationSelector::SocialWelfare => {
                let values = match input.welfare_stage {
                    SignedWelfareStage::PreTransfer => pre,
                    SignedWelfareStage::PostTransfer => &post,
                };
                weighted_sum(input.welfare_weights, values)?
            }
            SignedInformationSelector::Agent(agent) => post[agent],
        };
        let selected = match design.selected_policy {
            Some(declared) => policy == declared,
            None => scores.is_empty() || score > scores[winner],
        };
        if selected {
            winner = position;
            winning_post = post;
        }
        scores.push(score);
    }
    let selected_policy = design.policy_indices[winner];
    let pre_transfer = payoffs[selected_policy].clone();
    let social_pre_transfer = weighted_sum(input.welfare_weights, &pre_transfer)?;
    let social_post_transfer = weighted_sum(input.welfare_weights, &winning_post)?;
    Ok(SignedDesignEvaluation {
        selector_values: scores,
        selected_policy,
        ledger: SignedInformationLedger {
            pre_transfer,
            transfer,
            cost,
            post_transfer: winning_post,
        },
        social_pre_transfer,
        social_post_transfer,
        signed_by_agent: Vec::new(),
        signed_by_role: Vec::new(),
        signed_welfare: 0.0,
    })
}

fn design_accounts(
    agent_count: usize,
    design: &SignedInformationDesign<'_>,
) -> Result<(Vec<f64>, Vec<f64>), NumericalInputError> {
    let mut transfers = vec![0.0; agent_count];
    let mut costs = vec![0.0; agent_count];
    for &(payer, recipient, amount) in design.transfers {
        if payer >= agent_count
            || recipient >= agent_count
            || payer == recipient
            || !amount.is_finite()
            || amount < 0.0
        {
            return Err(NumericalInputError::invalid(
                "transfers",
                "Transfers need distinct declared agents and a finite nonnegative amount",
            ));
        }
        transfers[payer] = finite("transfers", transfers[payer] - amount)?;
        transfers[recipient] = finite("transfers", transfers[recipient] + amount)?;
    }
    for &(agent, amount) in design.costs {
        if agent >= agent_count || !amount.is_finite() || amount < 0.0 {
            return Err(NumericalInputError::invalid(
                "costs",
                "Costs need a declared agent and a finite nonnegative amount",
            ));
        }
        costs[agent] = finite("costs", costs[agent] + amount)?;
    }
    Ok((transfers, costs))
}

fn differences(current: &[f64], comparator: &[f64]) -> Result<Vec<f64>, NumericalInputError> {
    current
        .iter()
        .zip(comparator)
        .map(|(current, comparator)| finite("signed_values", current - comparator))
        .collect()
}

fn weighted_sum(weights: &[f64], values: &[f64]) -> Result<f64, NumericalInputError> {
    finite_sum(
        "welfare",
        weights
            .iter()
            .zip(values)
            .map(|(weight, value)| weight * value),
    )
}

fn finite_sum(
    field: &'static str,
    terms: impl IntoIterator<Item = f64>,
) -> Result<f64, NumericalInputError> {
    let arithmetic_error = || NumericalInputError::invalid(field, "Arithmetic must be finite");
    let total = terms
        .into_iter()
        .try_fold(BigRational::default(), |sum, term| {
            BigRational::from_float(term)
                .map(|value| sum + value)
                .ok_or_else(arithmetic_error)
        })?;
    round_to_binary64(&total).map_err(|_| arithmetic_error())
}

fn finite(field: &'static str, value: f64) -> Result<f64, NumericalInputError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NumericalInputError::invalid(
            field,
            "Arithmetic must be finite",
        ))
    }
}
