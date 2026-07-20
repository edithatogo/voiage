use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};

/// Validation failures for dimension-aware collections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainError {
    /// A collection has no values.
    Empty,
    /// A dimension has length zero.
    EmptyDimension {
        /// Zero-length dimension index.
        dimension: usize,
    },
    /// A nested collection has an inconsistent length.
    Ragged {
        /// Inconsistent dimension index.
        dimension: usize,
        /// Required dimension length.
        expected: usize,
        /// Observed dimension length.
        actual: usize,
        /// Nested collection index containing the mismatch.
        index: usize,
    },
    /// A numeric value is NaN or infinite.
    NonFinite {
        /// Flattened sample index.
        index: usize,
    },
    /// A strategy name is empty or whitespace only.
    BlankStrategy {
        /// Strategy index.
        index: usize,
    },
    /// A strategy name occurs more than once.
    DuplicateStrategy {
        /// Index of the repeated strategy.
        index: usize,
    },
}

impl fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid domain collection: {self:?}")
    }
}

impl std::error::Error for DomainError {}

/// A non-empty vector of finite sample values.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SampleVector(Vec<f64>);

impl SampleVector {
    /// Returns the number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the vector is empty. Valid vectors are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Borrows the validated values.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

impl TryFrom<Vec<f64>> for SampleVector {
    type Error = DomainError;

    fn try_from(values: Vec<f64>) -> Result<Self, Self::Error> {
        validate_values(&values)?;
        Ok(Self(values))
    }
}

impl<'de> Deserialize<'de> for SampleVector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<f64>::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

/// A non-empty rectangular matrix of finite sample values.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SampleMatrix(Vec<Vec<f64>>);

impl SampleMatrix {
    /// Returns `[rows, columns]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [self.0.len(), self.0[0].len()]
    }

    /// Borrows one row.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        self.0.get(row).map(Vec::as_slice)
    }

    /// Iterates over validated matrix rows.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        self.0.iter().map(Vec::as_slice)
    }

    /// Returns one value when both indices are in bounds.
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<f64> {
        self.0
            .get(row)
            .and_then(|values| values.get(column))
            .copied()
    }
}

impl TryFrom<Vec<Vec<f64>>> for SampleMatrix {
    type Error = DomainError;

    fn try_from(values: Vec<Vec<f64>>) -> Result<Self, Self::Error> {
        validate_matrix(&values)?;
        Ok(Self(values))
    }
}

impl<'de> Deserialize<'de> for SampleMatrix {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<Vec<f64>>::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

/// A non-empty rectangular cube of finite sample values.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SampleCube(Vec<Vec<Vec<f64>>>);

impl SampleCube {
    /// Returns `[planes, rows, columns]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        [self.0.len(), self.0[0].len(), self.0[0][0].len()]
    }

    /// Iterates over validated sample planes in input order.
    pub fn planes(&self) -> impl ExactSizeIterator<Item = &[Vec<f64>]> {
        self.0.iter().map(Vec::as_slice)
    }

    /// Returns one value when all indices are in bounds.
    #[must_use]
    pub fn get(&self, plane: usize, row: usize, column: usize) -> Option<f64> {
        self.0.get(plane)?.get(row)?.get(column).copied()
    }
}

impl TryFrom<Vec<Vec<Vec<f64>>>> for SampleCube {
    type Error = DomainError;

    fn try_from(values: Vec<Vec<Vec<f64>>>) -> Result<Self, Self::Error> {
        if values.is_empty() {
            return Err(DomainError::Empty);
        }
        let expected_rows = values[0].len();
        if expected_rows == 0 {
            return Err(DomainError::EmptyDimension { dimension: 1 });
        }
        let expected_columns = values[0][0].len();
        if expected_columns == 0 {
            return Err(DomainError::EmptyDimension { dimension: 2 });
        }
        let mut flat_index = 0;
        for (plane_index, plane) in values.iter().enumerate() {
            if plane.len() != expected_rows {
                return Err(DomainError::Ragged {
                    dimension: 1,
                    expected: expected_rows,
                    actual: plane.len(),
                    index: plane_index,
                });
            }
            for (row_index, row) in plane.iter().enumerate() {
                if row.len() != expected_columns {
                    return Err(DomainError::Ragged {
                        dimension: 2,
                        expected: expected_columns,
                        actual: row.len(),
                        index: row_index,
                    });
                }
                validate_finite(row, &mut flat_index)?;
            }
        }
        Ok(Self(values))
    }
}

impl<'de> Deserialize<'de> for SampleCube {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<Vec<Vec<f64>>>::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

/// An ordered, non-empty collection of unique non-blank strategy names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct StrategyCollection(Vec<String>);

impl StrategyCollection {
    /// Returns the number of strategies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the collection is empty. Valid collections are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Borrows the strategy at `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&str> {
        self.0.get(index).map(String::as_str)
    }

    /// Iterates over strategy names in declared order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

impl<T: Into<String>> TryFrom<Vec<T>> for StrategyCollection {
    type Error = DomainError;

    fn try_from(values: Vec<T>) -> Result<Self, Self::Error> {
        if values.is_empty() {
            return Err(DomainError::Empty);
        }
        let values: Vec<String> = values
            .into_iter()
            .map(Into::into)
            .map(|value: String| value.trim().to_owned())
            .collect();
        let mut seen = HashSet::with_capacity(values.len());
        for (index, value) in values.iter().enumerate() {
            if value.is_empty() {
                return Err(DomainError::BlankStrategy { index });
            }
            if !seen.insert(value.as_str()) {
                return Err(DomainError::DuplicateStrategy { index });
            }
        }
        Ok(Self(values))
    }
}

impl<'de> Deserialize<'de> for StrategyCollection {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<String>::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

fn validate_values(values: &[f64]) -> Result<(), DomainError> {
    if values.is_empty() {
        return Err(DomainError::Empty);
    }
    let mut index = 0;
    validate_finite(values, &mut index)
}

fn validate_matrix(values: &[Vec<f64>]) -> Result<(), DomainError> {
    if values.is_empty() {
        return Err(DomainError::Empty);
    }
    let expected = values[0].len();
    if expected == 0 {
        return Err(DomainError::EmptyDimension { dimension: 1 });
    }
    let mut flat_index = 0;
    for (index, row) in values.iter().enumerate() {
        if row.len() != expected {
            return Err(DomainError::Ragged {
                dimension: 1,
                expected,
                actual: row.len(),
                index,
            });
        }
        validate_finite(row, &mut flat_index)?;
    }
    Ok(())
}

fn validate_finite(values: &[f64], flat_index: &mut usize) -> Result<(), DomainError> {
    for value in values {
        if !value.is_finite() {
            return Err(DomainError::NonFinite { index: *flat_index });
        }
        *flat_index += 1;
    }
    Ok(())
}
