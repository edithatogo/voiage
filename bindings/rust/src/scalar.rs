//! Deterministic scalar-first VOI contract helpers.

const CONTRACT_VERSION: &str = "rust-core-scalar-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarError {
    EmptyRow,
    RaggedRows {
        expected: usize,
        actual: usize,
        row_index: usize,
    },
    NonFiniteInput,
    NegativeResearchCost,
    NonFiniteScalarInput,
}

impl ScalarError {
    pub fn message(&self) -> &'static str {
        match self {
            Self::EmptyRow => "net_benefits must contain non-empty rows",
            Self::RaggedRows { .. } => "net_benefits rows must have a consistent width",
            Self::NonFiniteInput => "net_benefits must contain only finite values",
            Self::NegativeResearchCost => "research_cost cannot be negative",
            Self::NonFiniteScalarInput => "inputs must be finite numbers",
        }
    }
}

impl core::fmt::Display for ScalarError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for ScalarError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ReportingEnvelope {
    pub contract_version: &'static str,
    pub method: &'static str,
    pub deterministic: bool,
    pub status: &'static str,
    pub policy: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvpiDiagnostics {
    pub sample_count: usize,
    pub strategy_count: usize,
    pub min_row_width: usize,
    pub max_row_width: usize,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvpiResult {
    pub value: f64,
    pub diagnostics: EvpiDiagnostics,
    pub reporting: ReportingEnvelope,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnbsDiagnostics {
    pub evsi_result: f64,
    pub research_cost: f64,
    pub finite: bool,
    pub non_negative_research_cost: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnbsResult {
    pub value: f64,
    pub diagnostics: EnbsDiagnostics,
    pub reporting: ReportingEnvelope,
}

fn reporting(method: &'static str, policy: &'static str) -> ReportingEnvelope {
    ReportingEnvelope {
        contract_version: CONTRACT_VERSION,
        method,
        deterministic: true,
        status: "complete",
        policy,
    }
}

fn validate_matrix(net_benefits: &[Vec<f64>]) -> Result<(usize, usize), ScalarError> {
    if net_benefits.is_empty() {
        return Ok((0, 0));
    }
    let first_width = net_benefits[0].len();
    if first_width == 0 {
        return Err(ScalarError::EmptyRow);
    }
    for (row_index, row) in net_benefits.iter().enumerate() {
        if row.len() != first_width {
            return Err(ScalarError::RaggedRows {
                expected: first_width,
                actual: row.len(),
                row_index,
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(ScalarError::NonFiniteInput);
        }
    }
    Ok((net_benefits.len(), first_width))
}

/// Contract-first EVPI result with deterministic diagnostics and reporting.
pub fn evpi_contract(net_benefits: &[Vec<f64>]) -> Result<EvpiResult, ScalarError> {
    let (sample_count, strategy_count) = validate_matrix(net_benefits)?;
    if sample_count == 0 || strategy_count <= 1 {
        return Ok(EvpiResult {
            value: 0.0,
            diagnostics: EvpiDiagnostics {
                sample_count,
                strategy_count,
                min_row_width: strategy_count,
                max_row_width: strategy_count,
                finite: true,
            },
            reporting: reporting("evpi", "sample_mean_vs_row_max"),
        });
    }

    let mut strategy_sums = vec![0.0; strategy_count];
    let mut max_row_sum = 0.0;

    for row in net_benefits {
        let mut row_max = row[0];
        for (index, value) in row.iter().enumerate() {
            strategy_sums[index] += value;
            if *value > row_max {
                row_max = *value;
            }
        }
        max_row_sum += row_max;
    }

    let sample_count_f = sample_count as f64;
    let max_expected = strategy_sums
        .iter()
        .map(|sum| sum / sample_count_f)
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_max = max_row_sum / sample_count_f;
    let value = (expected_max - max_expected).max(0.0);

    Ok(EvpiResult {
        value,
        diagnostics: EvpiDiagnostics {
            sample_count,
            strategy_count,
            min_row_width: strategy_count,
            max_row_width: strategy_count,
            finite: true,
        },
        reporting: reporting("evpi", "sample_mean_vs_row_max"),
    })
}

/// Scalar EVPI wrapper preserved for thin bindings.
pub fn evpi(net_benefits: &[Vec<f64>]) -> Result<f64, &'static str> {
    evpi_contract(net_benefits)
        .map(|result| result.value)
        .map_err(|error| error.message())
}

/// Contract-first ENBS result with deterministic diagnostics and reporting.
pub fn enbs_contract(evsi_result: f64, research_cost: f64) -> Result<EnbsResult, ScalarError> {
    if !evsi_result.is_finite() || !research_cost.is_finite() {
        return Err(ScalarError::NonFiniteScalarInput);
    }
    if research_cost < 0.0 {
        return Err(ScalarError::NegativeResearchCost);
    }

    Ok(EnbsResult {
        value: evsi_result - research_cost,
        diagnostics: EnbsDiagnostics {
            evsi_result,
            research_cost,
            finite: true,
            non_negative_research_cost: true,
        },
        reporting: reporting("enbs", "raw_subtraction"),
    })
}

/// Scalar ENBS wrapper preserved for thin bindings.
pub fn enbs(evsi_result: f64, research_cost: f64) -> Result<f64, &'static str> {
    enbs_contract(evsi_result, research_cost)
        .map(|result| result.value)
        .map_err(|error| error.message())
}
