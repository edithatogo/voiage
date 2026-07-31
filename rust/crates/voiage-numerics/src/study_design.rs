use std::cmp::Reverse;

use crate::NumericalInputError;

/// Binding-independent result for an enumerated COSS calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct CossKernelResult {
    /// Version of the native result contract.
    pub contract_version: &'static str,
    /// Stable identifier for the numerical procedure.
    pub estimator: &'static str,
    /// Signed expected net benefit of sampling in caller order.
    pub enbs: Vec<f64>,
    /// Caller-order indices of feasible designs.
    pub feasible_indices: Vec<usize>,
    /// Caller-order indices within the declared tie tolerance.
    pub tied_indices: Vec<usize>,
    /// Selected caller-order index, or `None` when no design is feasible.
    pub optimal_index: Option<usize>,
    /// Greatest feasible signed ENBS, or `None` when no design is feasible.
    pub maximum_enbs: Option<f64>,
    /// Whether the selected sample size is on the feasible-set boundary.
    pub boundary_state: &'static str,
}

/// Binding-independent result for the EVSI/EVPI efficiency diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct InformationEfficiencyKernelResult {
    /// Version of the native result contract.
    pub contract_version: &'static str,
    /// Unclamped EVSI/EVPI ratio, if EVPI is not numerically zero.
    pub ratio: Option<f64>,
    /// Stable bounds or zero-EVPI classification.
    pub status: &'static str,
    /// Value-scale tolerance used for bounds and numerical-zero checks.
    pub bound_tolerance: f64,
}

fn validate_tolerances(
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<(), NumericalInputError> {
    if !absolute_tolerance.is_finite() || absolute_tolerance < 0.0 {
        return Err(NumericalInputError::invalid(
            "absolute_tolerance",
            "absolute tolerance must be finite and non-negative",
        ));
    }
    if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
        return Err(NumericalInputError::invalid(
            "relative_tolerance",
            "relative tolerance must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_value(value: f64, field: &'static str) -> Result<(), NumericalInputError> {
    if !value.is_finite() || value < 0.0 {
        return Err(NumericalInputError::invalid(
            field,
            "values must be finite and non-negative",
        ));
    }
    Ok(())
}

fn select_optimum(tied_indices: &[usize], sample_sizes: &[u64], tie_policy: &str) -> Option<usize> {
    match tie_policy {
        "smallest_sample_size" => tied_indices
            .iter()
            .copied()
            .min_by_key(|&index| (sample_sizes[index], index)),
        "largest_sample_size" => tied_indices
            .iter()
            .copied()
            .min_by_key(|&index| (Reverse(sample_sizes[index]), index)),
        "first_declared" => tied_indices.first().copied(),
        _ => None,
    }
}

fn classify_boundary(
    optimal_index: usize,
    feasible_indices: &[usize],
    sample_sizes: &[u64],
) -> &'static str {
    let selected_size = sample_sizes[optimal_index];
    let (minimum_size, maximum_size) = feasible_indices.iter().fold(
        (selected_size, selected_size),
        |(minimum, maximum), &index| {
            (
                minimum.min(sample_sizes[index]),
                maximum.max(sample_sizes[index]),
            )
        },
    );
    if minimum_size == maximum_size {
        "both"
    } else if selected_size == minimum_size {
        "lower"
    } else if selected_size == maximum_size {
        "upper"
    } else {
        "interior"
    }
}

/// Computes a Curve of Optimal Sample Size result over enumerated designs.
///
/// Signed ENBS is retained for every design and only feasible designs
/// participate in selection. The returned vectors preserve caller order.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for empty or unequal vectors, non-finite or
/// negative values and tolerances, or an unknown tie policy.
pub fn coss(
    sample_sizes: &[u64],
    evsi_values: &[f64],
    research_costs: &[f64],
    feasible: &[bool],
    tie_policy: &str,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<CossKernelResult, NumericalInputError> {
    let design_count = sample_sizes.len();
    if design_count == 0 {
        return Err(NumericalInputError::invalid(
            "sample_sizes",
            "at least one evaluated design is required",
        ));
    }
    for (field, actual) in [
        ("evsi_values", evsi_values.len()),
        ("research_costs", research_costs.len()),
        ("feasible", feasible.len()),
    ] {
        if actual != design_count {
            return Err(NumericalInputError::dimension(
                field,
                design_count,
                actual,
                "all evaluated-design vectors must have equal length",
            ));
        }
    }
    validate_tolerances(absolute_tolerance, relative_tolerance)?;
    if !matches!(
        tie_policy,
        "smallest_sample_size" | "largest_sample_size" | "first_declared"
    ) {
        return Err(NumericalInputError::invalid(
            "tie_policy",
            "tie policy must be smallest_sample_size, largest_sample_size, or first_declared",
        ));
    }

    for &value in evsi_values {
        validate_value(value, "evsi_values")?;
    }
    for &value in research_costs {
        validate_value(value, "research_costs")?;
    }

    let enbs = evsi_values
        .iter()
        .zip(research_costs)
        .map(|(evsi, cost)| evsi - cost)
        .collect::<Vec<_>>();
    let feasible_indices = feasible
        .iter()
        .enumerate()
        .filter_map(|(index, &is_feasible)| is_feasible.then_some(index))
        .collect::<Vec<_>>();
    let Some(maximum_enbs) = feasible_indices
        .iter()
        .map(|&index| enbs[index])
        .max_by(f64::total_cmp)
    else {
        return Ok(CossKernelResult {
            contract_version: "1.0.0",
            estimator: "enumerated_signed_enbs",
            enbs,
            feasible_indices,
            tied_indices: Vec::new(),
            optimal_index: None,
            maximum_enbs: None,
            boundary_state: "none",
        });
    };

    let tie_tolerance = absolute_tolerance + relative_tolerance * maximum_enbs.abs().max(1.0);
    if !tie_tolerance.is_finite() {
        return Err(NumericalInputError::invalid(
            "relative_tolerance",
            "combined tie tolerance must be finite",
        ));
    }
    let tied_indices = feasible_indices
        .iter()
        .copied()
        .filter(|&index| maximum_enbs - enbs[index] <= tie_tolerance)
        .collect::<Vec<_>>();
    let Some(optimal_index) = select_optimum(&tied_indices, sample_sizes, tie_policy) else {
        return Err(NumericalInputError::invalid(
            "tie_policy",
            "tie policy did not select a feasible design",
        ));
    };
    let boundary_state = classify_boundary(optimal_index, &feasible_indices, sample_sizes);

    Ok(CossKernelResult {
        contract_version: "1.0.0",
        estimator: "enumerated_signed_enbs",
        enbs,
        feasible_indices,
        tied_indices,
        optimal_index: Some(optimal_index),
        maximum_enbs: Some(maximum_enbs),
        boundary_state,
    })
}

/// Computes the unclamped EVSI/EVPI information-efficiency diagnostic.
///
/// # Errors
///
/// Returns [`NumericalInputError`] for invalid tolerances, non-finite inputs,
/// materially negative EVPI, or EVSI outside the theoretical bounds by more
/// than the declared tolerance.
pub fn evsi_evpi_efficiency(
    expected_sample_information: f64,
    expected_perfect_information: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<InformationEfficiencyKernelResult, NumericalInputError> {
    validate_tolerances(absolute_tolerance, relative_tolerance)?;
    if !expected_sample_information.is_finite() {
        return Err(NumericalInputError::invalid("evsi", "EVSI must be finite"));
    }
    if !expected_perfect_information.is_finite() {
        return Err(NumericalInputError::invalid("evpi", "EVPI must be finite"));
    }
    let bound_tolerance =
        absolute_tolerance + relative_tolerance * expected_perfect_information.abs().max(1.0);
    if !bound_tolerance.is_finite() {
        return Err(NumericalInputError::invalid(
            "relative_tolerance",
            "combined bound tolerance must be finite",
        ));
    }
    if expected_perfect_information < -bound_tolerance {
        return Err(NumericalInputError::invalid(
            "evpi",
            "EVPI is materially negative",
        ));
    }
    if expected_perfect_information.abs() <= bound_tolerance {
        if expected_sample_information.abs() <= bound_tolerance {
            return Ok(InformationEfficiencyKernelResult {
                contract_version: "1.0.0",
                ratio: None,
                status: "undefined_zero_evpi",
                bound_tolerance,
            });
        }
        return Err(NumericalInputError::invalid(
            "evsi",
            "non-zero EVSI is inconsistent with numerically zero EVPI",
        ));
    }
    if expected_sample_information < -bound_tolerance
        || expected_sample_information > expected_perfect_information + bound_tolerance
    {
        return Err(NumericalInputError::invalid(
            "evsi",
            "EVSI is materially outside the theoretical interval from zero to EVPI",
        ));
    }
    let status = if expected_sample_information < 0.0 {
        "below_zero_within_tolerance"
    } else if expected_sample_information > expected_perfect_information {
        "above_one_within_tolerance"
    } else {
        "within_bounds"
    };
    Ok(InformationEfficiencyKernelResult {
        contract_version: "1.0.0",
        ratio: Some(expected_sample_information / expected_perfect_information),
        status,
        bound_tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coss_preserves_signed_curve_and_selects_only_feasible_designs() {
        let result = coss(
            &[100, 200, 300],
            &[5.0, 9.0, 10.0],
            &[8.0, 3.0, 12.0],
            &[true, false, true],
            "smallest_sample_size",
            0.0,
            0.0,
        )
        .unwrap();
        assert_eq!(result.enbs, vec![-3.0, 6.0, -2.0]);
        assert_eq!(result.feasible_indices, vec![0, 2]);
        assert_eq!(result.optimal_index, Some(2));
        assert_eq!(result.maximum_enbs, Some(-2.0));
        assert_eq!(result.boundary_state, "upper");
    }

    #[test]
    fn coss_applies_tolerance_and_deterministic_tie_policies() {
        let smallest = coss(
            &[300, 100, 200],
            &[10.0, 10.0, 10.0],
            &[0.0, 0.05, 0.02],
            &[true; 3],
            "smallest_sample_size",
            0.1,
            0.0,
        )
        .unwrap();
        assert_eq!(smallest.tied_indices, vec![0, 1, 2]);
        assert_eq!(smallest.optimal_index, Some(1));
        assert_eq!(smallest.boundary_state, "lower");

        let largest = coss(
            &[300, 100, 200],
            &[10.0, 10.0, 10.0],
            &[0.0, 0.05, 0.02],
            &[true; 3],
            "largest_sample_size",
            0.1,
            0.0,
        )
        .unwrap();
        assert_eq!(largest.optimal_index, Some(0));
        assert_eq!(largest.boundary_state, "upper");
    }

    #[test]
    fn coss_reports_no_optimum_when_every_design_is_infeasible() {
        let result = coss(
            &[100, 200],
            &[2.0, 5.0],
            &[1.0, 2.0],
            &[false, false],
            "first_declared",
            0.0,
            0.0,
        )
        .unwrap();
        assert_eq!(result.optimal_index, None);
        assert_eq!(result.maximum_enbs, None);
        assert_eq!(result.boundary_state, "none");
    }

    #[test]
    fn coss_accepts_duplicate_sizes_and_rejects_invalid_values() {
        let duplicate_sizes = coss(
            &[100, 100],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[true, true],
            "smallest_sample_size",
            2.0,
            0.0,
        )
        .unwrap();
        assert_eq!(duplicate_sizes.optimal_index, Some(0));
        assert_eq!(duplicate_sizes.boundary_state, "both");
        assert!(coss(
            &[100],
            &[f64::NAN],
            &[0.0],
            &[true],
            "first_declared",
            0.0,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn information_efficiency_preserves_near_bound_ratios() {
        let above = evsi_evpi_efficiency(10.05, 10.0, 0.1, 0.0).unwrap();
        assert_eq!(above.status, "above_one_within_tolerance");
        assert_eq!(above.ratio, Some(1.005_000_000_000_000_1));

        let below = evsi_evpi_efficiency(-0.05, 10.0, 0.1, 0.0).unwrap();
        assert_eq!(below.status, "below_zero_within_tolerance");
        assert_eq!(below.ratio, Some(-0.005));
    }

    #[test]
    fn information_efficiency_handles_zero_and_rejects_material_violations() {
        let zero = evsi_evpi_efficiency(0.0, 0.0, 1.0e-9, 0.0).unwrap();
        assert_eq!(zero.ratio, None);
        assert_eq!(zero.status, "undefined_zero_evpi");
        assert!(evsi_evpi_efficiency(1.0, 0.0, 1.0e-9, 0.0).is_err());
        assert!(evsi_evpi_efficiency(11.0, 10.0, 0.1, 0.0).is_err());
        assert!(evsi_evpi_efficiency(1.0, -10.0, 0.1, 0.0).is_err());
    }
}
