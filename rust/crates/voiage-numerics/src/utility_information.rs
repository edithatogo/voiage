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
    pub tie_set: Vec<String>,
    pub representative_action_id: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MeasureResult {
    pub status: String,
    pub value: Option<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct RootResult {
    pub status: String,
    pub estimate: Option<f64>,
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
    pub cross_problem_comparable: bool,
    pub required_shared_fields: Vec<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ExpectedUtilityInformationResult {
    pub schema_version: String,
    pub method: String,
    pub method_maturity: String,
    pub information_kind: String,
    pub current_expected_utility: f64,
    pub informed_expected_utility: f64,
    pub current_policy: PolicyResult,
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
        } if risk_aversion > 0.0
            && (risk_aversion - 1.0).abs() > f64::EPSILON
            && reference_wealth > 0.0
            && value > 0.0 =>
        {
            ((value / reference_wealth).powf(1.0 - risk_aversion) - 1.0) / (1.0 - risk_aversion)
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
    match *descriptor {
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
            let base = 1.0 + (1.0 - risk_aversion) * value;
            if base > 0.0 {
                Ok(reference_wealth * base.powf(1.0 / (1.0 - risk_aversion)))
            } else {
                Err(ExpectedUtilityError {
                    code: "utility_domain",
                })
            }
        }
        UtilityDescriptor::Exponential { .. } => Err(ExpectedUtilityError {
            code: "utility_domain",
        }),
    }
}

fn validate(
    input: &ExpectedUtilityInformationInput,
    descriptor: &UtilityDescriptor,
) -> Result<(), ExpectedUtilityError> {
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
    if input.information_cost_location != "ex_ante_sure_transfer" {
        return Err(ExpectedUtilityError {
            code: "invalid_cost_location",
        });
    }
    for row in &input.payoffs {
        for payoff in row {
            utility(descriptor, input.initial_wealth + payoff)?;
        }
    }
    Ok(())
}

fn choose(values: &[f64], ids: &[String]) -> (f64, PolicyResult) {
    let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut ties = values
        .iter()
        .zip(ids)
        .filter(|(v, _)| (**v - best).abs() <= 1e-12 * (1.0 + best.abs()))
        .map(|(_, id)| id.clone())
        .collect::<Vec<_>>();
    ties.sort();
    let representative_action_id = ties[0].clone();
    (
        best,
        PolicyResult {
            tie_set: ties,
            representative_action_id,
        },
    )
}

fn baseline(
    input: &ExpectedUtilityInformationInput,
    u: &UtilityDescriptor,
    transfer: f64,
) -> Result<(f64, PolicyResult), ExpectedUtilityError> {
    let values = (0..input.action_ids.len())
        .map(|a| {
            input
                .state_probabilities
                .iter()
                .enumerate()
                .map(|(s, p)| {
                    utility(u, input.initial_wealth + input.payoffs[s][a] + transfer).map(|x| p * x)
                })
                .sum()
        })
        .collect::<Result<Vec<f64>, _>>()?;
    Ok(choose(&values, &input.action_ids))
}

fn informed(
    input: &ExpectedUtilityInformationInput,
    u: &UtilityDescriptor,
    transfer: f64,
) -> Result<f64, ExpectedUtilityError> {
    input
        .information
        .signal_state_probabilities
        .iter()
        .map(|joint| {
            let values = (0..input.action_ids.len())
                .map(|a| {
                    joint
                        .iter()
                        .enumerate()
                        .try_fold(0.0, |total, (s, p)| {
                            if *p == 0.0 {
                                Ok(total)
                            } else {
                                utility(u, input.initial_wealth + input.payoffs[s][a] + transfer)
                                    .map(|value| total + p * value)
                            }
                        })
                        .unwrap_or(f64::NEG_INFINITY)
                })
                .collect::<Vec<_>>();
            if values.iter().all(|value| *value == f64::NEG_INFINITY) {
                return Err(ExpectedUtilityError {
                    code: "utility_domain",
                });
            }
            Ok(choose(&values, &input.action_ids).0)
        })
        .sum()
}

fn solve<F>(input: &ExpectedUtilityInformationInput, mut objective: F) -> RootResult
where
    F: FnMut(f64) -> Result<f64, ExpectedUtilityError>,
{
    let settings = &input.solver;
    let mut evaluations = 1usize;
    let Ok(zero) = objective(0.0) else {
        return RootResult {
            status: "utility_domain".into(),
            estimate: None,
        };
    };
    if zero.abs() <= settings.utility_tolerance {
        return RootResult {
            status: "zero_boundary".into(),
            estimate: Some(0.0),
        };
    }
    if settings.maximum_evaluations <= 1 {
        return RootResult {
            status: "max_evaluations".into(),
            estimate: None,
        };
    }
    let mut lower = 0.0;
    let mut upper = settings.initial_upper.min(settings.maximum_price);
    let mut upper_value;
    loop {
        evaluations += 1;
        upper_value = match objective(upper) {
            Ok(v) => v,
            Err(_) => {
                return RootResult {
                    status: "utility_domain".into(),
                    estimate: None,
                }
            }
        };
        if zero.signum() != upper_value.signum() || upper_value.abs() <= settings.utility_tolerance
        {
            break;
        }
        if evaluations >= settings.maximum_evaluations {
            return RootResult {
                status: "max_evaluations".into(),
                estimate: None,
            };
        }
        if upper >= settings.maximum_price {
            return RootResult {
                status: "not_bracketed".into(),
                estimate: None,
            };
        }
        upper = (upper * settings.expansion_factor).min(settings.maximum_price);
    }
    for _ in 0..settings.maximum_iterations {
        if evaluations >= settings.maximum_evaluations {
            return RootResult {
                status: "max_evaluations".into(),
                estimate: None,
            };
        }
        let estimate = f64::midpoint(lower, upper);
        evaluations += 1;
        let Ok(value) = objective(estimate) else {
            return RootResult {
                status: "utility_domain".into(),
                estimate: None,
            };
        };
        let tolerance =
            settings.absolute_price_tolerance + settings.relative_price_tolerance * estimate.abs();
        if upper - lower <= tolerance {
            return RootResult {
                status: "converged".into(),
                estimate: Some(estimate),
            };
        }
        if value.signum() == zero.signum() {
            lower = estimate;
        } else {
            upper = estimate;
        }
    }
    RootResult {
        status: "max_iterations".into(),
        estimate: None,
    }
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
    let i0 = informed(input, u, 0.0)?;
    let eui = i0 - b0;
    let cei = inverse(u, i0)? - inverse(u, b0)?;
    let bpi_root = solve(input, |price| Ok(informed(input, u, -price)? - b0));
    let spi_root = solve(input, |price| Ok(baseline(input, u, price)?.0 - i0));
    let ppi = input
        .terminal_outcome_floor
        .and_then(|floor| utility(u, floor).ok())
        .and_then(|floor_u| {
            let denominator = i0 - floor_u;
            if denominator > 0.0 {
                let value = eui / denominator;
                if (-1e-12..=1.0 + 1e-12).contains(&value) {
                    Some(value)
                } else {
                    None
                }
            } else {
                None
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
    Ok(ExpectedUtilityInformationResult {
        schema_version: "expected-utility-information-result-v1".into(),
        method: "expected_utility_information".into(),
        method_maturity: "experimental".into(),
        information_kind: input.information.kind.clone(),
        current_expected_utility: b0,
        informed_expected_utility: i0,
        current_policy,
        eui: measure("eui", Some(eui), None),
        cei: measure("cei", Some(cei), None),
        bpi: measure("bpi", bpi_root.estimate, Some(&bpi_root.status)),
        spi: measure("spi", spi_root.estimate, Some(&spi_root.status)),
        ppi: measure("ppi", ppi, None),
        bpi_root,
        spi_root,
        affine_reduction: affine,
        comparability: Comparability {
            stakeholder_scope_id: input.stakeholder_scope_id.clone(),
            cross_problem_comparable: false,
            required_shared_fields: vec![
                "stakeholder_scope_id".into(),
                "utility_identity".into(),
                "wealth_basis".into(),
            ],
        },
    })
}
