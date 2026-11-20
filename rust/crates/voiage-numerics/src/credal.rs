use std::collections::BTreeMap;

use voiage_domain::{
    AffineNetBenefit, BigRational, CredalContractError, CredalPoint, CredalPolytope,
    DominanceWitness, StrategyId,
};

use crate::NumericalInputError;

/// Verifies that a feasible point makes the proposed strategy optimal.
///
/// Comparisons use the exact affine objectives, including ties.
///
/// # Errors
/// Returns an error for missing or duplicate strategies or incompatible dimensions.
pub fn verify_e_admissibility(
    support: &CredalPolytope,
    strategies: &[AffineNetBenefit],
    strategy: &StrategyId,
    witness: &CredalPoint,
) -> Result<bool, NumericalInputError> {
    let objectives = index_strategies(support, strategies)?;
    let target = objectives.get(strategy).ok_or_else(unknown_strategy)?;
    if !support
        .contains_exact(witness)
        .map_err(|error| input_error("witness", error))?
    {
        return Ok(false);
    }
    let target_value = target
        .evaluate_exact(witness)
        .map_err(|error| input_error("witness", error))?;
    for other in objectives.values() {
        let value = other
            .evaluate_exact(witness)
            .map_err(|error| input_error("witness", error))?;
        if value > target_value {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Verifies a positive lower bound for a mixture of alternative strategies.
///
/// The supplied feasible point establishes nonempty support. Multipliers are
/// checked against the actual support and objectives. The residual linear term
/// is minimised over the bounding box, so approximate stationarity cannot
/// create a false proof of dominance.
///
/// # Errors
/// Returns an error for missing or duplicate strategies or incompatible dimensions.
pub fn verify_strict_credal_dominance(
    support: &CredalPolytope,
    strategies: &[AffineNetBenefit],
    feasible_point: &CredalPoint,
    witness: &DominanceWitness,
) -> Result<bool, NumericalInputError> {
    let objectives = index_strategies(support, strategies)?;
    let target = objectives
        .get(witness.strategy())
        .ok_or_else(unknown_strategy)?;
    let (_, target_intercept, target_coefficients) = target.objective();
    let (_, bounds) = support.coordinates();
    let inequalities = support.inequalities();
    let (alternatives, support_weights, margin) = witness.certificate();
    let expected_rows = inequalities.len() + 2 * bounds.len();
    if support_weights.len() != expected_rows {
        return Err(NumericalInputError::dimension(
            "support_multipliers",
            expected_rows,
            support_weights.len(),
            "The multiplier count must match the support rows",
        ));
    }

    let zero = BigRational::from_integer(0.into());
    let mut residual = vec![zero.clone(); bounds.len()];
    let mut lower = zero.clone();
    let mut total_weight = zero;
    for (strategy, &multiplier) in alternatives {
        let alternative = objectives.get(strategy).ok_or_else(unknown_strategy)?;
        let (_, intercept, coefficients) = alternative.objective();
        let weight = exact(multiplier)?;
        lower += &weight * (exact(intercept)? - exact(target_intercept)?);
        for ((term, &alt), &target) in residual
            .iter_mut()
            .zip(coefficients)
            .zip(target_coefficients)
        {
            *term += &weight * (exact(alt)? - exact(target)?);
        }
        total_weight += weight;
    }

    // For a support row a*x <= b, adding its nonnegative weighted residual
    // a*x - b gives a lower bound for the original mixture advantage.
    let (halfspace_weights, box_weights) = support_weights.split_at(inequalities.len());
    for (inequality, &multiplier) in inequalities.iter().zip(halfspace_weights) {
        let (coefficients, upper) = inequality.row();
        let weight = exact(multiplier)?;
        lower -= &weight * exact(upper)?;
        for (term, &coefficient) in residual.iter_mut().zip(coefficients) {
            *term += &weight * exact(coefficient)?;
        }
    }
    for ((term, bound), weights) in residual
        .iter_mut()
        .zip(bounds)
        .zip(box_weights.chunks_exact(2))
    {
        let (lower_bound, upper_bound) = bound.endpoints();
        let lo = exact(lower_bound)?;
        let hi = exact(upper_bound)?;
        let lower_weight = exact(weights[0])?;
        let upper_weight = exact(weights[1])?;
        lower += &lower_weight * &lo - &upper_weight * &hi;
        *term += upper_weight - lower_weight;
        lower += (&*term * lo).min(&*term * hi);
    }

    let is_nonempty = support
        .contains_exact(feasible_point)
        .map_err(|error| input_error("feasible_point", error))?;
    Ok(is_nonempty && lower >= exact(margin.get())? * total_weight)
}

fn index_strategies<'a>(
    support: &CredalPolytope,
    strategies: &'a [AffineNetBenefit],
) -> Result<BTreeMap<&'a StrategyId, &'a AffineNetBenefit>, NumericalInputError> {
    if strategies.is_empty() {
        return Err(NumericalInputError::invalid(
            "strategies",
            "A strategy is required",
        ));
    }
    let (parameters, _) = support.coordinates();
    let mut indexed = BTreeMap::new();
    for benefit in strategies {
        let (strategy, _, coefficients) = benefit.objective();
        if coefficients.len() != parameters.len() {
            return Err(NumericalInputError::dimension(
                "strategies",
                parameters.len(),
                coefficients.len(),
                "Each affine objective must match the support dimension",
            ));
        }
        if indexed.insert(strategy, benefit).is_some() {
            return Err(NumericalInputError::invalid(
                "strategies",
                "Strategy IDs must be unique",
            ));
        }
    }
    Ok(indexed)
}

fn unknown_strategy() -> NumericalInputError {
    NumericalInputError::invalid("strategy", "The certificate refers to an unknown strategy")
}

fn input_error(field: &'static str, error: CredalContractError) -> NumericalInputError {
    if let CredalContractError::DimensionMismatch { expected, actual } = error {
        NumericalInputError::dimension(
            field,
            expected,
            actual,
            "Witness dimension does not match the support",
        )
    } else {
        NumericalInputError::invalid(field, "Invalid credal parameter or affine objective")
    }
}

fn exact(value: f64) -> Result<BigRational, NumericalInputError> {
    BigRational::from_float(value).ok_or_else(|| {
        NumericalInputError::invalid(
            "certificate",
            "Credal coefficients and multipliers must be finite",
        )
    })
}
