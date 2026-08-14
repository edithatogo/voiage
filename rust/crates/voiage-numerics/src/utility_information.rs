//! Experimental expected-utility information-value and price kernel.
//!
//! The types are intentionally isolated from the stable scalar EVPI API while
//! the family remains under scientific review.
#![allow(missing_docs)]

#[derive(Clone, Debug, PartialEq)]
pub enum UtilityDescriptor {
    Affine {
        slope: f64,
        intercept: f64,
    },
    Exponential {
        risk_tolerance: f64,
        reference_wealth: f64,
    },
    Log {
        reference_wealth: f64,
    },
    Power {
        risk_aversion: f64,
        reference_wealth: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct InformationStructure {
    pub kind: String,
    pub signal_ids: Vec<String>,
    pub signal_state_probabilities: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolverSettings {
    pub initial_upper: f64,
    pub expansion_factor: f64,
    pub maximum_price: f64,
    pub absolute_price_tolerance: f64,
    pub relative_price_tolerance: f64,
    pub utility_tolerance: f64,
    pub maximum_iterations: usize,
    pub maximum_evaluations: usize,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            initial_upper: 1.0,
            expansion_factor: 2.0,
            maximum_price: 1.0e9,
            absolute_price_tolerance: 1.0e-10,
            relative_price_tolerance: 1.0e-10,
            utility_tolerance: 1.0e-12,
            maximum_iterations: 200,
            maximum_evaluations: 500,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedUtilityInformationInput {
    pub schema_version: String,
    pub decision_problem_id: String,
    pub stakeholder_scope_id: String,
    pub action_ids: Vec<String>,
    pub state_ids: Vec<String>,
    pub payoffs: Vec<Vec<f64>>,
    pub state_probabilities: Vec<f64>,
    pub initial_wealth: f64,
    pub payoff_unit: String,
    pub currency: Option<String>,
    pub price_date: Option<String>,
    pub information_cost_location: String,
    pub information: InformationStructure,
    pub terminal_outcome_floor: Option<f64>,
    pub solver: SolverSettings,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyResult {
    pub signal_id: Option<String>,
    pub tie_set: Vec<String>,
    pub representative_action_id: String,
    pub domain_exclusions: Vec<DomainExclusion>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct DomainExclusion {
    pub signal_id: Option<String>,
    pub action_id: String,
    pub state_ids: Vec<String>,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEvaluation {
    pub transfer: f64,
    pub objective_value: f64,
    pub policies: Vec<PolicyResult>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyTransition {
    pub transfer: f64,
    pub prior_policies: Vec<PolicyResult>,
    pub next_policies: Vec<PolicyResult>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MeasureResult {
    pub status: String,
    pub value: Option<f64>,
    pub diagnostic_code: Option<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct RootResult {
    pub status: String,
    pub estimate: Option<f64>,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub final_bracket_width: Option<f64>,
    pub residual: Option<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub lower_policies: Vec<PolicyResult>,
    pub upper_policies: Vec<PolicyResult>,
    pub evaluated_policies: Vec<PolicyEvaluation>,
    pub policy_switched: bool,
    pub transitions: Vec<PolicyTransition>,
    pub termination_reason: String,
    pub solver: SolverSettings,
}
#[derive(Clone, Debug, PartialEq)]
pub struct AffineReduction {
    pub status: String,
    pub monetary_measure: Option<String>,
    pub value: Option<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Comparability {
    pub stakeholder_scope_id: String,
    pub numeric_within_problem: bool,
    pub numeric_cross_problem: bool,
    pub required_shared_fields: Vec<String>,
    pub ranking_equivalence: Vec<RankingEquivalence>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct RankingEquivalence {
    pub scope: String,
    pub left_measure: String,
    pub right_measure: String,
    pub status: String,
    pub condition: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedUtilityInformationResult {
    pub schema_version: String,
    pub method: String,
    pub method_maturity: String,
    pub information_kind: String,
    pub current_expected_utility: f64,
    pub informed_expected_utility: f64,
    pub current_certainty_equivalent: f64,
    pub informed_certainty_equivalent: f64,
    pub current_policy: PolicyResult,
    pub informed_policies: Vec<PolicyResult>,
    pub eui: MeasureResult,
    pub cei: MeasureResult,
    pub bpi: MeasureResult,
    pub spi: MeasureResult,
    pub ppi: MeasureResult,
    pub bpi_root: RootResult,
    pub spi_root: RootResult,
    pub affine_reduction: AffineReduction,
    pub comparability: Comparability,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedUtilityError {
    code: &'static str,
}
impl ExpectedUtilityError {
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }
}
impl core::fmt::Display for ExpectedUtilityError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.code)
    }
}
impl std::error::Error for ExpectedUtilityError {}

fn utility(descriptor: &UtilityDescriptor, value: f64) -> Result<f64, ExpectedUtilityError> {
    let result = match *descriptor {
        UtilityDescriptor::Affine { slope, intercept } if slope > 0.0 => {
            slope.mul_add(value, intercept)
        }
        UtilityDescriptor::Exponential {
            risk_tolerance,
            reference_wealth,
        } if risk_tolerance > 0.0 => -(-(value - reference_wealth) / risk_tolerance).exp(),
        UtilityDescriptor::Log { reference_wealth } if reference_wealth > 0.0 && value > 0.0 => {
            (value / reference_wealth).ln()
        }
        UtilityDescriptor::Power {
            risk_aversion,
            reference_wealth,
        } if risk_aversion > 0.0 && reference_wealth > 0.0 && value > 0.0 => {
            let log_ratio = (value / reference_wealth).ln();
            let exponent = 1.0 - risk_aversion;
            if exponent == 0.0 {
                log_ratio
            } else {
                (exponent * log_ratio).exp_m1() / exponent
            }
        }
        _ => {
            return Err(ExpectedUtilityError {
                code: "utility_domain",
            })
        }
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ExpectedUtilityError {
            code: "utility_domain",
        })
    }
}

fn inverse(descriptor: &UtilityDescriptor, value: f64) -> Result<f64, ExpectedUtilityError> {
    let result = match *descriptor {
        UtilityDescriptor::Affine { slope, intercept } => Ok((value - intercept) / slope),
        UtilityDescriptor::Exponential {
            risk_tolerance,
            reference_wealth,
        } if value < 0.0 => Ok(reference_wealth - risk_tolerance * (-value).ln()),
        UtilityDescriptor::Log { reference_wealth } => Ok(reference_wealth * value.exp()),
        UtilityDescriptor::Power {
            risk_aversion,
            reference_wealth,
        } => {
            let exponent = 1.0 - risk_aversion;
            let scaled = exponent * value;
            if exponent == 0.0 {
                Ok(reference_wealth * value.exp())
            } else if scaled > -1.0 {
                Ok(reference_wealth * (scaled.ln_1p() / exponent).exp())
            } else {
                Err(ExpectedUtilityError {
                    code: "utility_domain",
                })
            }
        }
        UtilityDescriptor::Exponential { .. } => Err(ExpectedUtilityError {
            code: "utility_domain",
        }),
    }?;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ExpectedUtilityError {
            code: "utility_domain",
        })
    }
}

#[allow(clippy::too_many_lines)] // Keep the frozen v1 validation order auditable in one fail-closed gate.
fn validate(
    input: &ExpectedUtilityInformationInput,
    descriptor: &UtilityDescriptor,
) -> Result<(), ExpectedUtilityError> {
    let unique_nonempty = |ids: &[String]| {
        let set = ids.iter().collect::<std::collections::BTreeSet<_>>();
        !ids.is_empty() && set.len() == ids.len() && ids.iter().all(|id| !id.trim().is_empty())
    };
    if input.schema_version != "expected-utility-information-input-v1"
        || input.decision_problem_id.is_empty()
        || input.stakeholder_scope_id.is_empty()
        || input.payoff_unit.is_empty()
        || !unique_nonempty(&input.action_ids)
        || !unique_nonempty(&input.state_ids)
        || !unique_nonempty(&input.information.signal_ids)
        || !matches!(
            input.information.kind.as_str(),
            "clairvoyant" | "finite_signal"
        )
        || !input.initial_wealth.is_finite()
        || input
            .payoffs
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(ExpectedUtilityError {
            code: "invalid_input",
        });
    }
    if input.currency.is_some() != input.price_date.is_some() {
        return Err(ExpectedUtilityError {
            code: "invalid_input",
        });
    }
    if input.currency.as_ref().is_some_and(|currency| {
        currency.len() != 3 || !currency.bytes().all(|byte| byte.is_ascii_uppercase())
    }) || input.price_date.as_ref().is_some_and(|date| {
        date.len() != 10
            || !date.bytes().enumerate().all(|(index, byte)| {
                matches!(index, 4 | 7) && byte == b'-'
                    || !matches!(index, 4 | 7) && byte.is_ascii_digit()
            })
    }) {
        return Err(ExpectedUtilityError {
            code: "invalid_input",
        });
    }
    let descriptor_valid = match *descriptor {
        UtilityDescriptor::Affine { slope, intercept } => {
            slope.is_finite() && slope > 0.0 && intercept.is_finite()
        }
        UtilityDescriptor::Exponential {
            risk_tolerance,
            reference_wealth,
        } => risk_tolerance.is_finite() && risk_tolerance > 0.0 && reference_wealth.is_finite(),
        UtilityDescriptor::Log { reference_wealth } => {
            reference_wealth.is_finite() && reference_wealth > 0.0
        }
        UtilityDescriptor::Power {
            risk_aversion,
            reference_wealth,
        } => {
            risk_aversion.is_finite()
                && risk_aversion > 0.0
                && reference_wealth.is_finite()
                && reference_wealth > 0.0
        }
    };
    if !descriptor_valid {
        return Err(ExpectedUtilityError {
            code: "invalid_utility",
        });
    }
    let states = input.state_ids.len();
    let actions = input.action_ids.len();
    if states == 0
        || actions == 0
        || input.payoffs.len() != states
        || input.payoffs.iter().any(|row| row.len() != actions)
    {
        return Err(ExpectedUtilityError {
            code: "dimension_mismatch",
        });
    }
    if input.state_probabilities.len() != states
        || input
            .state_probabilities
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0)
        || (input.state_probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        return Err(ExpectedUtilityError {
            code: "invalid_probability",
        });
    }
    if input.information.signal_state_probabilities.len() != input.information.signal_ids.len()
        || input
            .information
            .signal_state_probabilities
            .iter()
            .any(|row| row.len() != states)
    {
        return Err(ExpectedUtilityError {
            code: "dimension_mismatch",
        });
    }
    if input
        .information
        .signal_state_probabilities
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0)
        || (input
            .information
            .signal_state_probabilities
            .iter()
            .flatten()
            .sum::<f64>()
            - 1.0)
            .abs()
            > 1e-10
    {
        return Err(ExpectedUtilityError {
            code: "invalid_probability",
        });
    }
    for state in 0..states {
        let marginal = input
            .information
            .signal_state_probabilities
            .iter()
            .map(|row| row[state])
            .sum::<f64>();
        if (marginal - input.state_probabilities[state]).abs() > 1e-10 {
            return Err(ExpectedUtilityError {
                code: "invalid_probability",
            });
        }
    }
    if input.information.kind == "clairvoyant"
        && (input.information.signal_ids.len() != states
            || input
                .information
                .signal_state_probabilities
                .iter()
                .enumerate()
                .any(|(signal, row)| {
                    row.iter().enumerate().any(|(state, value)| {
                        let expected = if signal == state {
                            input.state_probabilities[state]
                        } else {
                            0.0
                        };
                        (*value - expected).abs() > 1e-10
                    })
                }))
    {
        return Err(ExpectedUtilityError {
            code: "invalid_clairvoyance",
        });
    }
    if input.information_cost_location != "ex_ante_sure_transfer" {
        return Err(ExpectedUtilityError {
            code: "invalid_cost_location",
        });
    }
    let solver = &input.solver;
    if !solver.initial_upper.is_finite()
        || solver.initial_upper <= 0.0
        || !solver.expansion_factor.is_finite()
        || solver.expansion_factor <= 1.0
        || !solver.maximum_price.is_finite()
        || solver.maximum_price < solver.initial_upper
        || !solver.absolute_price_tolerance.is_finite()
        || solver.absolute_price_tolerance <= 0.0
        || !solver.relative_price_tolerance.is_finite()
        || solver.relative_price_tolerance <= 0.0
        || !solver.utility_tolerance.is_finite()
        || solver.utility_tolerance <= 0.0
        || solver.maximum_iterations == 0
        || solver.maximum_evaluations == 0
    {
        return Err(ExpectedUtilityError {
            code: "invalid_solver",
        });
    }
    if let Some(floor) = input.terminal_outcome_floor {
        if !floor.is_finite()
            || input
                .state_probabilities
                .iter()
                .enumerate()
                .filter(|(_, probability)| **probability > 0.0)
                .flat_map(|(state, _)| {
                    input.payoffs[state]
                        .iter()
                        .map(move |payoff| input.initial_wealth + payoff)
                })
                .any(|outcome| floor > outcome)
        {
            return Err(ExpectedUtilityError {
                code: "invalid_ppi_anchor",
            });
        }
        utility(descriptor, floor).map_err(|_| ExpectedUtilityError {
            code: "invalid_ppi_anchor",
        })?;
    }
    Ok(())
}

fn choose(
    values: &[(usize, f64)],
    ids: &[String],
    mut domain_exclusions: Vec<DomainExclusion>,
) -> Result<(f64, PolicyResult), ExpectedUtilityError> {
    if values.is_empty() {
        return Err(ExpectedUtilityError {
            code: "utility_domain",
        });
    }
    let best = values
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut ties = values
        .iter()
        .filter(|(_, value)| (*value - best).abs() <= 1e-12 * (1.0 + best.abs()))
        .map(|(index, _)| ids[*index].clone())
        .collect::<Vec<_>>();
    ties.sort();
    let representative_action_id = ties[0].clone();
    for exclusion in &mut domain_exclusions {
        exclusion.state_ids.sort();
    }
    domain_exclusions.sort_by(|left, right| {
        left.signal_id
            .cmp(&right.signal_id)
            .then(left.action_id.cmp(&right.action_id))
            .then(left.state_ids.cmp(&right.state_ids))
    });
    Ok((
        best,
        PolicyResult {
            signal_id: None,
            tie_set: ties,
            representative_action_id,
            domain_exclusions,
        },
    ))
}

fn sort_policies(policies: &mut [PolicyResult]) {
    policies.sort_by(|left, right| {
        left.signal_id
            .cmp(&right.signal_id)
            .then(
                left.representative_action_id
                    .cmp(&right.representative_action_id),
            )
            .then(left.tie_set.cmp(&right.tie_set))
    });
}

fn baseline(
    input: &ExpectedUtilityInformationInput,
    u: &UtilityDescriptor,
    transfer: f64,
) -> Result<(f64, PolicyResult), ExpectedUtilityError> {
    let mut values = Vec::new();
    let mut exclusions = Vec::new();
    for a in 0..input.action_ids.len() {
        let mut failed_states = Vec::new();
        let value = input
            .state_probabilities
            .iter()
            .enumerate()
            .try_fold(0.0, |total, (s, p)| {
                if *p == 0.0 {
                    Ok(total)
                } else {
                    match utility(u, input.initial_wealth + input.payoffs[s][a] + transfer) {
                        Ok(value) => Ok(total + p * value),
                        Err(error) => {
                            failed_states.push(input.state_ids[s].clone());
                            Err(error)
                        }
                    }
                }
            });
        match value {
            Ok(value) => values.push((a, value)),
            Err(_) => exclusions.push(DomainExclusion {
                signal_id: None,
                action_id: input.action_ids[a].clone(),
                state_ids: failed_states,
                reason: "utility_domain".into(),
            }),
        }
    }
    choose(&values, &input.action_ids, exclusions)
}

fn informed(
    input: &ExpectedUtilityInformationInput,
    u: &UtilityDescriptor,
    transfer: f64,
) -> Result<(f64, Vec<PolicyResult>), ExpectedUtilityError> {
    let evaluated = input
        .information
        .signal_state_probabilities
        .iter()
        .enumerate()
        .map(|(signal, joint)| {
            if joint.iter().all(|probability| *probability == 0.0) {
                return Ok((
                    0.0,
                    PolicyResult {
                        signal_id: Some(input.information.signal_ids[signal].clone()),
                        tie_set: Vec::new(),
                        representative_action_id: String::new(),
                        domain_exclusions: Vec::new(),
                    },
                ));
            }
            let mut values = Vec::new();
            let mut exclusions = Vec::new();
            for a in 0..input.action_ids.len() {
                let mut failed_states = Vec::new();
                let value = joint.iter().enumerate().try_fold(0.0, |total, (s, p)| {
                    if *p == 0.0 {
                        Ok(total)
                    } else {
                        match utility(u, input.initial_wealth + input.payoffs[s][a] + transfer) {
                            Ok(value) => Ok(total + p * value),
                            Err(error) => {
                                failed_states.push(input.state_ids[s].clone());
                                Err(error)
                            }
                        }
                    }
                });
                match value {
                    Ok(value) => values.push((a, value)),
                    Err(_) => exclusions.push(DomainExclusion {
                        signal_id: Some(input.information.signal_ids[signal].clone()),
                        action_id: input.action_ids[a].clone(),
                        state_ids: failed_states,
                        reason: "utility_domain".into(),
                    }),
                }
            }
            let (value, mut policy) = choose(&values, &input.action_ids, exclusions)?;
            policy.signal_id = Some(input.information.signal_ids[signal].clone());
            Ok((value, policy))
        })
        .collect::<Result<Vec<_>, ExpectedUtilityError>>()?;
    let total = evaluated.iter().map(|(value, _)| value).sum();
    let mut policies = evaluated
        .into_iter()
        .map(|(_, policy)| policy)
        .collect::<Vec<_>>();
    sort_policies(&mut policies);
    Ok((total, policies))
}

#[allow(clippy::too_many_lines)] // The bounded state machine is intentionally linear for auditability.
fn solve<F>(input: &ExpectedUtilityInformationInput, mut objective: F) -> RootResult
where
    F: FnMut(f64) -> Result<(f64, Vec<PolicyResult>), ExpectedUtilityError>,
{
    let settings = input.solver.clone();
    let mut history: Vec<PolicyEvaluation> = Vec::new();
    let evaluation_count = std::cell::Cell::new(0usize);
    let mut evaluate = |transfer: f64| {
        evaluation_count.set(evaluation_count.get() + 1);
        let (value, mut policies) = objective(transfer)?;
        sort_policies(&mut policies);
        history.push(PolicyEvaluation {
            transfer,
            objective_value: value,
            policies: policies.clone(),
        });
        Ok::<_, ExpectedUtilityError>((value, policies))
    };
    let finish = |status: &str,
                  estimate,
                  lower: Option<f64>,
                  upper: Option<f64>,
                  residual,
                  iterations,
                  history: Vec<PolicyEvaluation>| {
        let mut history = history;
        history.sort_by(|left, right| left.transfer.total_cmp(&right.transfer));
        let transitions = history
            .windows(2)
            .filter(|pair| {
                pair[0].policies.len() != pair[1].policies.len()
                    || pair[0]
                        .policies
                        .iter()
                        .zip(&pair[1].policies)
                        .any(|(prior, next)| {
                            prior.signal_id != next.signal_id
                                || prior.tie_set != next.tie_set
                                || prior.representative_action_id != next.representative_action_id
                        })
            })
            .map(|pair| PolicyTransition {
                transfer: pair[1].transfer,
                prior_policies: pair[0].policies.clone(),
                next_policies: pair[1].policies.clone(),
            })
            .collect::<Vec<_>>();
        let policies_at = |bound: Option<f64>| {
            bound
                .and_then(|value| {
                    history
                        .iter()
                        .rev()
                        .find(|item| item.transfer.to_bits() == value.to_bits())
                        .map(|item| item.policies.clone())
                })
                .unwrap_or_default()
        };
        RootResult {
            status: status.into(),
            estimate,
            lower,
            upper,
            final_bracket_width: lower.zip(upper).map(|(a, b)| b - a),
            residual,
            iterations,
            evaluations: evaluation_count.get(),
            lower_policies: policies_at(lower),
            upper_policies: policies_at(upper),
            evaluated_policies: history,
            policy_switched: !transitions.is_empty(),
            transitions,
            termination_reason: status.into(),
            solver: settings.clone(),
        }
    };
    let Ok((zero, _)) = evaluate(0.0) else {
        return finish("utility_domain", None, None, None, None, 0, history);
    };
    if zero.abs() <= settings.utility_tolerance {
        return finish(
            "zero_boundary",
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(zero),
            0,
            history,
        );
    }
    if settings.maximum_evaluations <= 1 {
        return finish("max_evaluations", None, Some(0.0), None, None, 0, history);
    }
    let mut lower = 0.0;
    let mut upper = settings.initial_upper.min(settings.maximum_price);
    loop {
        if evaluation_count.get() >= settings.maximum_evaluations {
            return finish(
                "max_evaluations",
                None,
                Some(lower),
                Some(upper),
                None,
                0,
                history,
            );
        }
        let Ok((value, _)) = evaluate(upper) else {
            return finish(
                "utility_domain",
                None,
                Some(lower),
                Some(upper),
                None,
                0,
                history,
            );
        };
        if value.abs() <= settings.utility_tolerance {
            return finish(
                "converged",
                Some(upper),
                Some(upper),
                Some(upper),
                Some(value),
                0,
                history,
            );
        }
        if zero.signum() != value.signum() {
            break;
        }
        if upper >= settings.maximum_price {
            return finish(
                "not_bracketed",
                None,
                Some(lower),
                Some(upper),
                None,
                0,
                history,
            );
        }
        upper = (upper * settings.expansion_factor).min(settings.maximum_price);
    }
    for iteration in 1..=settings.maximum_iterations {
        if evaluation_count.get() >= settings.maximum_evaluations {
            return finish(
                "max_evaluations",
                None,
                Some(lower),
                Some(upper),
                None,
                iteration - 1,
                history,
            );
        }
        let estimate = f64::midpoint(lower, upper);
        if estimate.to_bits() == lower.to_bits() || estimate.to_bits() == upper.to_bits() {
            return finish(
                "discontinuous_no_root",
                None,
                Some(lower),
                Some(upper),
                None,
                iteration,
                history,
            );
        }
        let Ok((value, _)) = evaluate(estimate) else {
            return finish(
                "utility_domain",
                None,
                Some(lower),
                Some(upper),
                None,
                iteration,
                history,
            );
        };
        if upper - lower
            <= settings.absolute_price_tolerance
                + settings.relative_price_tolerance * estimate.abs()
            && value.abs() <= settings.utility_tolerance
        {
            return finish(
                "converged",
                Some(estimate),
                Some(lower),
                Some(upper),
                Some(value),
                iteration,
                history,
            );
        }
        if value.signum() == zero.signum() {
            lower = estimate;
        } else {
            upper = estimate;
        }
    }
    finish(
        "max_iterations",
        None,
        Some(lower),
        Some(upper),
        None,
        settings.maximum_iterations,
        history,
    )
}

fn measure(name: &str, value: Option<f64>, root_status: Option<&str>) -> MeasureResult {
    let status = if value.is_some() {
        "available"
    } else if root_status.is_some() {
        "failed"
    } else {
        "unavailable"
    };
    let _ = name;
    MeasureResult {
        status: status.into(),
        value,
        diagnostic_code: None,
    }
}

#[allow(clippy::too_many_lines)] // Keep the frozen pairwise matrix explicit and reviewable.
fn comparability(
    input: &ExpectedUtilityInformationInput,
    utility: &UtilityDescriptor,
) -> Comparability {
    let affine = matches!(utility, UtilityDescriptor::Affine { .. });
    let translation_invariant = matches!(
        utility,
        UtilityDescriptor::Affine { .. } | UtilityDescriptor::Exponential { .. }
    );
    let rule =
        |scope: &str, left: &str, right: &str, status: &str, condition: &str| RankingEquivalence {
            scope: scope.into(),
            left_measure: left.into(),
            right_measure: right.into(),
            status: status.into(),
            condition: condition.into(),
        };
    Comparability {
        stakeholder_scope_id: input.stakeholder_scope_id.clone(),
        numeric_within_problem: false,
        numeric_cross_problem: false,
        required_shared_fields: vec![
            "currency".into(),
            "information_convention".into(),
            "payoff_unit".into(),
            "price_date".into(),
            "stakeholder_scope_id".into(),
            "utility_identity".into(),
            "wealth_basis".into(),
        ],
        ranking_equivalence: vec![
            rule(
                "within_problem",
                "eui",
                "cei",
                "equivalent",
                "same declared utility",
            ),
            rule(
                "within_problem",
                "eui",
                "spi",
                "equivalent",
                "same declared utility",
            ),
            rule(
                "within_problem",
                "eui",
                "ppi",
                "equivalent",
                "same valid fixed floor",
            ),
            rule(
                "within_problem",
                "eui",
                "bpi",
                if translation_invariant {
                    "equivalent"
                } else {
                    "not_assured"
                },
                "affine or exponential utility",
            ),
            rule(
                "within_problem",
                "cei",
                "spi",
                "equivalent",
                "same declared utility",
            ),
            rule(
                "cross_problem",
                "eui",
                "cei",
                if affine { "conditional" } else { "not_assured" },
                "affine utility and all required shared fields",
            ),
            rule(
                "cross_problem",
                "eui",
                "bpi",
                if affine { "conditional" } else { "not_assured" },
                "affine utility and all required shared fields",
            ),
            rule(
                "cross_problem",
                "cei",
                "bpi",
                if translation_invariant {
                    "conditional"
                } else {
                    "not_assured"
                },
                "affine or exponential utility and all required shared fields",
            ),
            rule(
                "cross_problem",
                "cei",
                "spi",
                "conditional",
                "common monetary and stakeholder basis",
            ),
            rule(
                "cross_problem",
                "eui",
                "ppi",
                "conditional",
                "same valid floor, utility identity and normalization",
            ),
        ],
    }
}

/// Calculate expected-utility information values and indifference prices.
///
/// # Errors
///
/// Returns a stable input error when dimensions, probabilities, utility
/// domains, or the cost-location contract are invalid.
pub fn expected_utility_information(
    input: &ExpectedUtilityInformationInput,
    u: &UtilityDescriptor,
) -> Result<ExpectedUtilityInformationResult, ExpectedUtilityError> {
    validate(input, u)?;
    let (b0, current_policy) = baseline(input, u, 0.0)?;
    let (i0, informed_policies) = informed(input, u, 0.0)?;
    let eui = i0 - b0;
    let current_certainty_equivalent = inverse(u, b0)?;
    let informed_certainty_equivalent = inverse(u, i0)?;
    let cei = informed_certainty_equivalent - current_certainty_equivalent;
    let bpi_root = solve(input, |price| {
        let (value, policies) = informed(input, u, -price)?;
        Ok((value - b0, policies))
    });
    let spi_root = solve(input, |price| {
        let (value, policy) = baseline(input, u, price)?;
        Ok((value - i0, vec![policy]))
    });
    let (ppi, ppi_diagnostic) =
        input
            .terminal_outcome_floor
            .map_or((None, Some("ppi_floor_missing".into())), |floor| {
                let Ok(floor_u) = utility(u, floor) else {
                    return (None, Some("ppi_anchor_invalid".into()));
                };
                let denominator = i0 - floor_u;
                if denominator > 0.0 {
                    let value = eui / denominator;
                    if (-1e-12..=1.0 + 1e-12).contains(&value) {
                        (Some(value), None)
                    } else {
                        (None, Some("ppi_out_of_bounds".into()))
                    }
                } else {
                    (None, Some("ppi_nonpositive_denominator".into()))
                }
            });
    let affine = if let UtilityDescriptor::Affine { slope, .. } = *u {
        AffineReduction {
            status: "available".into(),
            monetary_measure: Some(
                if input.information.kind == "clairvoyant" {
                    "evpi"
                } else {
                    "evsi"
                }
                .into(),
            ),
            value: Some(eui / slope),
        }
    } else {
        AffineReduction {
            status: "unavailable".into(),
            monetary_measure: None,
            value: None,
        }
    };
    let mut ppi_measure = measure("ppi", ppi, None);
    ppi_measure.diagnostic_code = ppi_diagnostic;
    Ok(ExpectedUtilityInformationResult {
        schema_version: "expected-utility-information-result-v1".into(),
        method: "expected_utility_information".into(),
        method_maturity: "experimental".into(),
        information_kind: input.information.kind.clone(),
        current_expected_utility: b0,
        informed_expected_utility: i0,
        current_certainty_equivalent,
        informed_certainty_equivalent,
        current_policy,
        informed_policies,
        eui: measure("eui", Some(eui), None),
        cei: measure("cei", Some(cei), None),
        bpi: measure("bpi", bpi_root.estimate, Some(&bpi_root.status)),
        spi: measure("spi", spi_root.estimate, Some(&spi_root.status)),
        ppi: ppi_measure,
        bpi_root,
        spi_root,
        affine_reduction: affine,
        comparability: comparability(input, u),
    })
}
