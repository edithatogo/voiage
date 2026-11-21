use std::f64::consts::{PI, SQRT_2};

use voiage_domain::{Identifier, SampleMatrix};

use crate::{NumericalInputError, ObservationChannel};

/// Gaussian measurement parameters for one latent state and acquisition action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TobitState {
    mean: f64,
    standard_deviation: f64,
    detection_probability: f64,
}

impl TobitState {
    /// Constructs a Gaussian signal and the probability of obtaining a measurement.
    /// Detection precedes left-censoring, conditional on the state and action.
    ///
    /// # Errors
    /// Returns an error for a nonfinite mean, a nonpositive or nonfinite standard
    /// deviation, or a detection probability outside the closed unit interval.
    pub fn new(
        mean: f64,
        standard_deviation: f64,
        detection_probability: f64,
    ) -> Result<Self, NumericalInputError> {
        if !mean.is_finite() {
            return Err(NumericalInputError::invalid(
                "mean",
                "Signal mean must be finite",
            ));
        }
        if !standard_deviation.is_finite() || standard_deviation <= 0.0 {
            return Err(NumericalInputError::invalid(
                "standard_deviation",
                "Signal standard deviation must be finite and positive",
            ));
        }
        if !(0.0..=1.0).contains(&detection_probability) {
            return Err(NumericalInputError::invalid(
                "detection_probability",
                "Detection probability must be between zero and one",
            ));
        }
        Ok(Self {
            mean,
            standard_deviation,
            detection_probability,
        })
    }

    fn standardize(self, value: f64) -> f64 {
        difference_in_units(value, self.mean, self.standard_deviation)
    }
}

/// Recorded outcome of one Gaussian measurement subject to a lower limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TobitObservation {
    /// The acquisition produced no measurement.
    Unavailable,
    /// The acquisition reported that the measured signal lay below the limit.
    BelowLimit,
    /// The acquisition supplied the continuous signal at or above the limit.
    Reading(f64),
}

/// Left-censored Gaussian observations for one acquisition action.
#[derive(Clone, Debug, PartialEq)]
pub struct TobitModel {
    limit: f64,
    states: Vec<TobitState>,
}

impl TobitModel {
    /// Defines the action's lower limit and signal parameters in latent state order.
    ///
    /// # Errors
    /// Returns an error for a nonfinite limit or an empty state list.
    pub fn new(limit: f64, states: Vec<TobitState>) -> Result<Self, NumericalInputError> {
        if !limit.is_finite() {
            return Err(NumericalInputError::invalid(
                "limit",
                "Quantification limit must be finite",
            ));
        }
        if states.is_empty() {
            return Err(NumericalInputError::invalid(
                "states",
                "At least one latent state is required",
            ));
        }
        Ok(Self { limit, states })
    }

    /// Returns one log likelihood per state in declaration order.
    /// Censored and unavailable reports use probability masses. A measured
    /// reading uses its Gaussian density times the detection probability.
    /// An impossible observation has log likelihood of negative infinity.
    ///
    /// # Errors
    /// Returns an error for a nonfinite reading or a reading below the limit.
    pub fn log_likelihoods(
        &self,
        observation: TobitObservation,
    ) -> Result<Vec<f64>, NumericalInputError> {
        if let TobitObservation::Reading(value) = observation {
            if !value.is_finite() || value < self.limit {
                return Err(NumericalInputError::invalid(
                    "observation",
                    "A reading must be finite and at or above the quantification limit",
                ));
            }
        }
        Ok(self
            .states
            .iter()
            .map(|state| {
                let detection = state.detection_probability;
                match observation {
                    TobitObservation::Unavailable => (-detection).ln_1p(),
                    TobitObservation::BelowLimit => {
                        detection.ln() + standard_normal_log_cdf(state.standardize(self.limit))
                    }
                    TobitObservation::Reading(value) => {
                        detection.ln() + standard_normal_log_density(state.standardize(value))
                            - state.standard_deviation.ln()
                    }
                }
            })
            .collect())
    }

    /// Constructs a finite EVSI channel for a declared partition of the observations.
    /// Outcome IDs name unavailability, censoring, and the successive measured
    /// bins in that order. The bins start at the limit, split at the cutpoints,
    /// and end with the unbounded upper bin. The sequential kernel evaluates
    /// this partition using the normalized binary64 likelihood weights.
    ///
    /// # Errors
    /// Returns an error for nonfinite or unordered cutpoints, cutpoints at or
    /// below the limit, invalid outcome IDs, or a positive bin weight that
    /// cannot be represented in binary64.
    pub fn finite_channel(
        &self,
        action: Identifier,
        cutpoints: &[f64],
        outcomes: Vec<Identifier>,
    ) -> Result<ObservationChannel, NumericalInputError> {
        let mut previous = self.limit;
        for &cutpoint in cutpoints {
            if !cutpoint.is_finite() || cutpoint <= previous {
                return Err(NumericalInputError::invalid(
                    "cutpoints",
                    "Cutpoints must increase strictly above the quantification limit",
                ));
            }
            previous = cutpoint;
        }
        let rows = self
            .states
            .iter()
            .map(|&state| self.bin_weights(state, cutpoints))
            .collect::<Result<Vec<_>, _>>()?;
        let weights = SampleMatrix::try_from(rows).map_err(|_| {
            NumericalInputError::invalid("observation_weights", "Invalid Gaussian bin weights")
        })?;
        ObservationChannel::from_weights(action, outcomes, &weights)
    }

    fn bin_weights(
        &self,
        state: TobitState,
        cutpoints: &[f64],
    ) -> Result<Vec<f64>, NumericalInputError> {
        let detection = state.detection_probability;
        if detection == 0.0 {
            let mut weights = vec![0.0; cutpoints.len() + 3];
            weights[0] = 1.0;
            return Ok(weights);
        }
        let log_detection = detection.ln();
        let mut lower_endpoint = self.limit;
        let mut lower = state.standardize(lower_endpoint);
        let mut weights = Vec::with_capacity(cutpoints.len() + 3);
        weights.push(1.0 - detection);
        weights.push(bin_weight(log_detection + standard_normal_log_cdf(lower))?);
        for &cutpoint in cutpoints {
            let upper = state.standardize(cutpoint);
            let difference = cutpoint - lower_endpoint;
            let log_width = if difference.is_finite() {
                difference.ln() - state.standard_deviation.ln()
            } else {
                difference_in_units(cutpoint, lower_endpoint, state.standard_deviation).ln()
            };
            weights.push(bin_weight(
                log_detection + standard_normal_log_interval(lower, upper, log_width),
            )?);
            lower_endpoint = cutpoint;
            lower = upper;
        }
        weights.push(bin_weight(log_detection + standard_normal_log_cdf(-lower))?);
        Ok(weights)
    }
}

fn difference_in_units(end: f64, start: f64, scale: f64) -> f64 {
    let difference = end - start;
    if difference.is_finite() {
        difference / scale
    } else {
        end / scale - start / scale
    }
}

fn standard_normal_log_density(value: f64) -> f64 {
    (-0.5 * value) * value - 0.5 * (2.0 * PI).ln()
}

fn standard_normal_log_cdf(value: f64) -> f64 {
    if value < -36.0 {
        // The Mills ratio expansion retains the log of a Gaussian left tail
        // below binary64 probability range. Six terms suffice at this cutoff.
        let inverse_square = value.recip().powi(2);
        let correction = inverse_square
            * (-1.0
                + inverse_square
                    * (3.0
                        + inverse_square
                            * (-15.0
                                + inverse_square
                                    * (105.0
                                        + inverse_square * (-945.0 + inverse_square * 10_395.0)))));
        standard_normal_log_density(value) - (-value).ln() + correction.ln_1p()
    } else if value <= 0.0 {
        (0.5 * libm::erfc(-value / SQRT_2)).ln()
    } else {
        (-0.5 * libm::erfc(value / SQRT_2)).ln_1p()
    }
}

fn standard_normal_log_interval(lower: f64, upper: f64, log_width: f64) -> f64 {
    let midpoint = lower.midpoint(upper);
    if midpoint.is_finite() && log_width + midpoint.abs().ln_1p() <= 1e-3_f64.ln() {
        // Integrate the Gaussian density expansion at the midpoint. Keeping
        // the quadratic term gives (midpoint² - 1) * width² / 24. The first
        // omitted term is fourth order in width * (1 + |midpoint|).
        let width = log_width.exp();
        let quadratic = ((midpoint * width).powi(2) - width.powi(2)) / 24.0;
        return log_width + standard_normal_log_density(midpoint) + quadratic.ln_1p();
    }
    let (large, small) = if lower >= 0.0 {
        (
            standard_normal_log_cdf(-lower),
            standard_normal_log_cdf(-upper),
        )
    } else {
        (
            standard_normal_log_cdf(upper),
            standard_normal_log_cdf(lower),
        )
    };
    large + (-(small - large).exp_m1()).ln()
}

fn bin_weight(log_weight: f64) -> Result<f64, NumericalInputError> {
    let weight = log_weight.exp();
    if weight.is_finite() && weight > 0.0 {
        return Ok(weight);
    }
    Err(NumericalInputError::invalid(
        "observation_weights",
        "A Gaussian bin weight cannot be represented in binary64",
    ))
}
