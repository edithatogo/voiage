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

/// Discounting, recruitment-delay, and time-horizon parameters for population ENBS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudyScheduleConfig {
    /// Annual continuous discount rate (`>= 0.0`).
    pub discount_rate: f64,
    /// Recruitment and reporting delay in years (`>= 0.0`).
    pub recruitment_delay_years: f64,
    /// Decision time horizon in years (`> 0.0`).
    pub time_horizon_years: f64,
    /// Incident eligible population per year (`> 0.0`).
    pub incident_population_per_year: f64,
}

/// Computes discounted population ENBS accounting for recruitment delay and time horizon.
///
/// Effective discounted beneficiaries over `[recruitment_delay_years, time_horizon_years]`
/// equals `0.0` when `recruitment_delay_years >= time_horizon_years`, yielding `-research_cost`.
///
/// # Errors
///
/// Returns [`NumericalInputError`] when inputs are non-finite or violate domain bounds.
pub fn discounted_population_enbs(
    per_patient_evsi: f64,
    research_cost: f64,
    schedule: &StudyScheduleConfig,
) -> Result<f64, NumericalInputError> {
    if !per_patient_evsi.is_finite() || per_patient_evsi < 0.0 {
        return Err(NumericalInputError::invalid(
            "per_patient_evsi",
            "per-patient EVSI must be finite and non-negative",
        ));
    }
    if !research_cost.is_finite() || research_cost < 0.0 {
        return Err(NumericalInputError::invalid(
            "research_cost",
            "research cost must be finite and non-negative",
        ));
    }
    if !schedule.discount_rate.is_finite() || schedule.discount_rate < 0.0 {
        return Err(NumericalInputError::invalid(
            "discount_rate",
            "discount rate must be finite and non-negative",
        ));
    }
    if !schedule.recruitment_delay_years.is_finite() || schedule.recruitment_delay_years < 0.0 {
        return Err(NumericalInputError::invalid(
            "recruitment_delay_years",
            "recruitment delay must be finite and non-negative",
        ));
    }
    if !schedule.time_horizon_years.is_finite() || schedule.time_horizon_years <= 0.0 {
        return Err(NumericalInputError::invalid(
            "time_horizon_years",
            "time horizon must be finite and positive",
        ));
    }
    if !schedule.incident_population_per_year.is_finite()
        || schedule.incident_population_per_year <= 0.0
    {
        return Err(NumericalInputError::invalid(
            "incident_population_per_year",
            "incident population per year must be finite and positive",
        ));
    }

    let effective_population = if schedule.recruitment_delay_years >= schedule.time_horizon_years {
        0.0
    } else if schedule.discount_rate == 0.0 {
        schedule.incident_population_per_year
            * (schedule.time_horizon_years - schedule.recruitment_delay_years)
    } else {
        let r = schedule.discount_rate;
        let factor = ((-r * schedule.recruitment_delay_years).exp()
            - (-r * schedule.time_horizon_years).exp())
            / r;
        schedule.incident_population_per_year * factor
    };

    Ok(per_patient_evsi * effective_population - research_cost)
}
