use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use voiage_domain::{BigRational, Identifier, SampleMatrix};

use crate::NumericalInputError;

/// A finite, action-specific conditional observation distribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationChannel {
    id: Identifier,
    outcomes: Vec<Identifier>,
    likelihoods: Vec<Vec<BigRational>>,
}

impl ObservationChannel {
    /// Constructs the channel from nonnegative relative weights per latent state.
    ///
    /// The rows follow the declared state order. The columns follow `outcomes`.
    /// Each row is normalized exactly using the supplied binary64 values.
    ///
    /// # Errors
    /// Returns an error for duplicate or mismatched outcome IDs, negative weights,
    /// or a row with no positive weight.
    pub fn from_weights(
        id: Identifier,
        outcomes: Vec<Identifier>,
        weights: &SampleMatrix,
    ) -> Result<Self, NumericalInputError> {
        check_outcomes(&outcomes, weights.shape()[1])?;
        let likelihoods = weights
            .rows()
            .map(|row| normalize(row, "observation_weights"))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            id,
            outcomes,
            likelihoods,
        })
    }

    /// Returns the acquisition action ID.
    #[must_use]
    pub fn id(&self) -> &Identifier {
        &self.id
    }

    /// Returns the outcome IDs in column order.
    #[must_use]
    pub fn outcomes(&self) -> &[Identifier] {
        &self.outcomes
    }

    /// Returns exact likelihoods in state-major, outcome-minor order.
    #[must_use]
    pub fn likelihoods(&self) -> &[Vec<BigRational>] {
        &self.likelihoods
    }

    /// Applies a state-independent random transformation to the channel outcomes.
    /// The transition rows follow the original outcome order. The columns follow
    /// the supplied new outcome IDs. The acquisition action keeps its identity.
    ///
    /// # Errors
    /// Returns an error for incompatible dimensions, duplicate outcome IDs,
    /// negative weights, or a transition row with no positive weight.
    pub fn garble(
        &self,
        outcomes: Vec<Identifier>,
        transition_weights: &SampleMatrix,
    ) -> Result<Self, NumericalInputError> {
        let [sources, destinations] = transition_weights.shape();
        check_length("garbling", self.outcomes.len(), sources)?;
        check_outcomes(&outcomes, destinations)?;
        let kernel = transition_weights
            .rows()
            .map(|row| normalize(row, "garbling"))
            .collect::<Result<Vec<_>, _>>()?;
        let likelihoods = self
            .likelihoods
            .iter()
            .map(|source| {
                let mut result = vec![BigRational::default(); destinations];
                for (probability, transition) in source.iter().zip(&kernel) {
                    for (total, weight) in result.iter_mut().zip(transition) {
                        *total += probability * weight;
                    }
                }
                result
            })
            .collect();
        Ok(Self {
            id: self.id.clone(),
            outcomes,
            likelihoods,
        })
    }
}

/// Result of a bounded sequential experiment over a finite latent state space.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequentialEvsi {
    /// The experiment has been evaluated exactly.
    Complete {
        /// Best expected terminal net benefit before observing any data.
        current_value: BigRational,
        /// Expected terminal net benefit under optimal sequential acquisition.
        expected_value: BigRational,
        /// Expected value of the experiment above the current decision value.
        evsi: BigRational,
        /// Number of outcome branches considered, including impossible outcomes.
        branches_examined: usize,
    },
    /// The branch budget was exhausted before the experiment could be evaluated.
    Indeterminate {
        /// Best expected terminal net benefit before observing any data.
        current_value: BigRational,
        /// Number of outcome branches considered.
        branches_examined: usize,
    },
}

/// Computes EVSI for a finite sequence of acquisition decisions.
///
/// Supply the same latent state order in `prior_weights`, `net_benefits`, and
/// every channel. Net benefits are `[state][terminal strategy]`. Acquisition
/// actions change the conditional observation distribution. The latent state
/// and terminal payoffs are fixed, observations are independent conditional on
/// that state and the selected action, and the current observation history can
/// determine the next action. The decision maker can stop at any stage. This
/// measures gross information value for the specified acquisition actions and
/// common terminal payoffs.
///
/// Nonnegative relative weights are normalized exactly. The kernel propagates
/// unnormalized joint probabilities and maximizes each history's contribution
/// to terminal value. A complete calculation therefore has nonnegative EVSI and
/// preserves Blackwell order under state-independent garbling of corresponding
/// channels with the same acquisition options and payoffs.
///
/// # Errors
/// Returns an error for invalid prior weights, inconsistent state counts, an
/// empty acquisition stage or repeated action IDs in a stage.
pub fn sequential_evsi(
    prior_weights: &[f64],
    net_benefits: &SampleMatrix,
    stages: &[Vec<ObservationChannel>],
    max_branches: NonZeroUsize,
) -> Result<SequentialEvsi, NumericalInputError> {
    let [latent_count, _] = net_benefits.shape();
    check_length("prior_weights", latent_count, prior_weights.len())?;
    for stage in stages {
        let mut actions = BTreeSet::new();
        if stage.is_empty() {
            return Err(NumericalInputError::invalid(
                "stages",
                "Each acquisition stage needs an action",
            ));
        }
        for action in stage {
            check_length(
                "observation_weights",
                latent_count,
                action.likelihoods.len(),
            )?;
            if !actions.insert(action.id()) {
                return Err(NumericalInputError::invalid(
                    "stages",
                    "Acquisition action IDs must be unique within a stage",
                ));
            }
        }
    }

    let prior = normalize(prior_weights, "prior_weights")?;
    let benefits = net_benefits
        .rows()
        .map(|row| {
            row.iter()
                .map(|&value| exact(value, "net_benefits"))
                .collect()
        })
        .collect::<Result<Vec<Vec<_>>, _>>()?;
    let current_value = terminal_value(&prior, &benefits);
    let mut branches_examined = 0;
    let root = History::new(0, prior, current_value.clone());
    match evaluate(
        stages,
        &benefits,
        root,
        max_branches,
        &mut branches_examined,
    ) {
        Some(expected_value) => Ok(SequentialEvsi::Complete {
            evsi: &expected_value - &current_value,
            current_value,
            expected_value,
            branches_examined,
        }),
        None => Ok(SequentialEvsi::Indeterminate {
            current_value,
            branches_examined,
        }),
    }
}

struct History {
    stage: usize,
    joint: Vec<BigRational>,
    best: BigRational,
    action: usize,
    outcome: usize,
    action_value: BigRational,
}

impl History {
    fn new(stage: usize, joint: Vec<BigRational>, stop_value: BigRational) -> Self {
        Self {
            stage,
            joint,
            best: stop_value,
            action: 0,
            outcome: 0,
            action_value: BigRational::default(),
        }
    }
}

fn evaluate(
    stages: &[Vec<ObservationChannel>],
    benefits: &[Vec<BigRational>],
    root: History,
    max_branches: NonZeroUsize,
    examined: &mut usize,
) -> Option<BigRational> {
    let zero = BigRational::default();
    // Keep only the active depth-first path, with a running value for each
    // acquisition action. The call stack stays bounded independently of depth.
    let mut path = vec![root];
    loop {
        let history = path.last_mut()?;
        let channel = stages
            .get(history.stage)
            .and_then(|stage| stage.get(history.action));
        let Some(channel) = channel else {
            let value = path.pop()?.best;
            if let Some(parent) = path.last_mut() {
                parent.action_value += value;
                continue;
            }
            return Some(value);
        };
        if history.outcome == channel.outcomes.len() {
            let value = std::mem::take(&mut history.action_value);
            if value > history.best {
                history.best = value;
            }
            history.action += 1;
            history.outcome = 0;
            continue;
        }
        if *examined == max_branches.get() {
            return None;
        }
        *examined += 1;
        let outcome = history.outcome;
        history.outcome += 1;
        let joint: Vec<_> = history
            .joint
            .iter()
            .zip(&channel.likelihoods)
            .map(|(weight, row)| weight * &row[outcome])
            .collect();
        if joint.iter().any(|probability| probability != &zero) {
            let stop_value = terminal_value(&joint, benefits);
            let next = History::new(history.stage + 1, joint, stop_value);
            path.push(next);
        }
    }
}

fn terminal_value(joint: &[BigRational], benefits: &[Vec<BigRational>]) -> BigRational {
    let mut totals = vec![BigRational::default(); benefits[0].len()];
    for (probability, row) in joint.iter().zip(benefits) {
        for (value, payoff) in totals.iter_mut().zip(row) {
            *value += probability * payoff;
        }
    }
    totals.into_iter().max().unwrap_or_default()
}

fn normalize(
    weights: &[f64],
    field: &'static str,
) -> Result<Vec<BigRational>, NumericalInputError> {
    let zero = BigRational::default();
    let mut probabilities = weights
        .iter()
        .map(|&weight| exact(weight, field))
        .collect::<Result<Vec<_>, _>>()?;
    if probabilities.iter().any(|value| value < &zero) {
        return Err(NumericalInputError::invalid(
            field,
            "Relative weights must be nonnegative",
        ));
    }
    let total: BigRational = probabilities.iter().sum();
    if total == zero {
        return Err(NumericalInputError::invalid(
            field,
            "Relative weights need a positive total",
        ));
    }
    for probability in &mut probabilities {
        *probability /= &total;
    }
    Ok(probabilities)
}

fn exact(value: f64, field: &'static str) -> Result<BigRational, NumericalInputError> {
    BigRational::from_float(value)
        .ok_or_else(|| NumericalInputError::invalid(field, "Values must be finite"))
}

fn check_outcomes(outcomes: &[Identifier], columns: usize) -> Result<(), NumericalInputError> {
    check_length("outcomes", columns, outcomes.len())?;
    if outcomes.iter().collect::<BTreeSet<_>>().len() != outcomes.len() {
        return Err(NumericalInputError::invalid(
            "outcomes",
            "Outcome IDs must be unique",
        ));
    }
    Ok(())
}

fn check_length(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), NumericalInputError> {
    if expected != actual {
        return Err(NumericalInputError::dimension(
            field,
            expected,
            actual,
            "Lengths must match declared states or outcomes",
        ));
    }
    Ok(())
}
