use crate::NumericalInputError;

/// Supported interpretations of the willingness-to-pay input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WtpMode {
    /// One threshold applies to every cost/effect pair.
    Scalar,
    /// Every threshold applies to every cost/effect pair.
    Thresholds,
    /// Each sample has its own vector of thresholds.
    SampleThresholds {
        /// Number of uncertainty samples.
        sample_count: usize,
        /// Number of thresholds owned by each sample.
        threshold_count: usize,
    },
    /// A time-bounded compatibility mode for elementwise WTP arrays.
    LegacyElementwise,
}

/// Rust-owned net-benefit values in row-major output order.
#[derive(Clone, Debug, PartialEq)]
pub struct NetBenefitKernelResult {
    /// Calculated values in row-major output order.
    pub values: Vec<f64>,
}

/// Calculate `effect * willingness_to_pay - cost`.
///
/// Negative finite thresholds are valid. All inputs and results must be finite.
pub fn net_benefit(
    costs: &[f64],
    effects: &[f64],
    willingness_to_pay: &[f64],
    mode: WtpMode,
) -> Result<NetBenefitKernelResult, NumericalInputError> {
    if costs.is_empty() {
        return Err(NumericalInputError::invalid(
            "costs",
            "costs and effects must not be empty",
        ));
    }
    if costs.len() != effects.len() {
        return Err(NumericalInputError::dimension(
            "effects",
            costs.len(),
            effects.len(),
            "costs and effects must have the same number of values",
        ));
    }
    if willingness_to_pay.is_empty() {
        return Err(NumericalInputError::invalid(
            "willingness_to_pay",
            "willingness to pay must not be empty",
        ));
    }
    if costs
        .iter()
        .chain(effects)
        .chain(willingness_to_pay)
        .any(|value| !value.is_finite())
    {
        return Err(NumericalInputError::invalid(
            "net_benefit",
            "costs, effects, and willingness to pay must be finite",
        ));
    }

    let mut values = Vec::new();
    let mut push_value = |value_index: usize, threshold: f64| {
        let value = effects[value_index] * threshold - costs[value_index];
        if value.is_finite() {
            values.push(value);
            Ok(())
        } else {
            Err(NumericalInputError::invalid(
                "net_benefit",
                "net benefit result is not finite",
            ))
        }
    };

    match mode {
        WtpMode::Scalar => {
            if willingness_to_pay.len() != 1 {
                return Err(NumericalInputError::dimension(
                    "willingness_to_pay",
                    1,
                    willingness_to_pay.len(),
                    "scalar willingness to pay must contain one value",
                ));
            }
            for value_index in 0..costs.len() {
                push_value(value_index, willingness_to_pay[0])?;
            }
        }
        WtpMode::Thresholds => {
            for value_index in 0..costs.len() {
                for threshold in willingness_to_pay {
                    push_value(value_index, *threshold)?;
                }
            }
        }
        WtpMode::SampleThresholds {
            sample_count,
            threshold_count,
        } => {
            if sample_count == 0
                || threshold_count == 0
                || costs.len() % sample_count != 0
                || willingness_to_pay.len() != sample_count * threshold_count
            {
                return Err(NumericalInputError::invalid(
                    "willingness_to_pay",
                    "sample-specific thresholds do not match the sample dimension",
                ));
            }
            let values_per_sample = costs.len() / sample_count;
            for sample in 0..sample_count {
                for within_sample in 0..values_per_sample {
                    let value_index = sample * values_per_sample + within_sample;
                    for threshold in 0..threshold_count {
                        push_value(
                            value_index,
                            willingness_to_pay[sample * threshold_count + threshold],
                        )?;
                    }
                }
            }
        }
        WtpMode::LegacyElementwise => {
            if willingness_to_pay.len() != costs.len() {
                return Err(NumericalInputError::dimension(
                    "willingness_to_pay",
                    costs.len(),
                    willingness_to_pay.len(),
                    "elementwise willingness to pay must match costs and effects",
                ));
            }
            for (value_index, threshold) in willingness_to_pay.iter().enumerate() {
                push_value(value_index, *threshold)?;
            }
        }
    }
    Ok(NetBenefitKernelResult { values })
}

#[cfg(test)]
mod tests {
    use super::{net_benefit, WtpMode};

    #[test]
    fn scalar_and_negative_thresholds_are_supported() {
        let result = net_benefit(&[100.0, 150.0], &[0.5, 0.6], &[-5_000.0], WtpMode::Scalar)
            .expect("finite inputs");
        assert_eq!(result.values, vec![-2_600.0, -3_150.0]);
    }

    #[test]
    fn threshold_axis_is_row_major() {
        let result = net_benefit(
            &[100.0, 150.0],
            &[0.5, 0.6],
            &[10_000.0, 20_000.0],
            WtpMode::Thresholds,
        )
        .expect("finite inputs");
        assert_eq!(result.values, vec![4_900.0, 9_900.0, 5_850.0, 11_850.0]);
    }

    #[test]
    fn sample_specific_thresholds_preserve_sample_ownership() {
        let result = net_benefit(
            &[100.0, 150.0, 90.0, 140.0],
            &[0.5, 0.6, 0.45, 0.55],
            &[10_000.0, 20_000.0, 15_000.0, 25_000.0],
            WtpMode::SampleThresholds {
                sample_count: 2,
                threshold_count: 2,
            },
        )
        .expect("matching samples");
        let expected = [
            4_900.0, 9_900.0, 5_850.0, 11_850.0, 6_660.0, 11_160.0, 8_110.0, 13_610.0,
        ];
        assert!(result
            .values
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-9));
    }

    #[test]
    fn non_finite_input_and_overflow_fail_closed() {
        assert!(net_benefit(&[f64::NAN], &[1.0], &[1.0], WtpMode::Scalar).is_err());
        assert!(net_benefit(&[0.0], &[f64::MAX], &[2.0], WtpMode::Scalar).is_err());
    }
}
