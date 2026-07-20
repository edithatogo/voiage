use crate::NumericalInputError;

/// Computes raw expected net benefit of sampling as `EVSI - research cost`.
///
/// Negative results are valid and are not clipped by this binding-independent
/// kernel.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when either input is non-finite or when the
/// research cost is negative.
pub fn enbs(evsi_result: f64, research_cost: f64) -> Result<f64, NumericalInputError> {
    if !evsi_result.is_finite() {
        return Err(NumericalInputError::invalid(
            "evsi_result",
            "EVSI result must be finite",
        ));
    }
    if !research_cost.is_finite() {
        return Err(NumericalInputError::invalid(
            "research_cost",
            "research cost must be finite",
        ));
    }
    if research_cost < 0.0 {
        return Err(NumericalInputError::invalid(
            "research_cost",
            "research cost must not be negative",
        ));
    }
    Ok(evsi_result - research_cost)
}
