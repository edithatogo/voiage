use core::cmp::Ordering;
use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::{Identifier, SampleVector, StrategyId, Threshold};

/// Invalid inputs to a credal decision contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CredalContractError {
    /// The parameter or strategy list was empty.
    Empty,
    /// A parameter or strategy identifier was repeated.
    DuplicateIdentifier,
    /// The input had a different number of coordinates or multipliers.
    DimensionMismatch {
        /// Number of entries required by the contract.
        expected: usize,
        /// Number of entries supplied.
        actual: usize,
    },
    /// An endpoint, coefficient calculation or other quantity was not finite.
    NonFinite,
    /// An interval's lower endpoint exceeded its upper endpoint.
    ReversedInterval,
    /// The single point in the bounding box violated a support inequality.
    InfeasibleSingleton,
    /// A rational parameter coordinate had a zero denominator.
    InvalidRationalCoordinate,
    /// A certificate contained malformed or degenerate multipliers.
    InvalidMultipliers,
    /// The stated normalized certificate margin was nonpositive or malformed.
    InvalidMargin,
}

impl fmt::Display for CredalContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid credal contract: {self:?}")
    }
}

impl std::error::Error for CredalContractError {}

/// Finite, closed support for one parameter coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CredalInterval {
    lower: f64,
    upper: f64,
}

impl CredalInterval {
    /// Constructs an ordered finite interval.
    ///
    /// # Errors
    /// Returns an error for a nonfinite endpoint or reversed bounds.
    pub fn new(lower: f64, upper: f64) -> Result<Self, CredalContractError> {
        finite(lower)?;
        finite(upper)?;
        if lower > upper {
            return Err(CredalContractError::ReversedInterval);
        }
        Ok(Self { lower, upper })
    }

    /// Returns the lower and upper endpoints in that order.
    #[must_use]
    pub const fn endpoints(self) -> (f64, f64) {
        (self.lower, self.upper)
    }
}

/// A nonempty parameter vector with exact rational coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredalPoint(Vec<BigRational>);

impl CredalPoint {
    /// Constructs a point with reduced rational coordinates.
    ///
    /// # Errors
    /// Returns an error for an empty point or a zero denominator.
    pub fn new(coordinates: Vec<BigRational>) -> Result<Self, CredalContractError> {
        if coordinates.is_empty() {
            return Err(CredalContractError::Empty);
        }
        coordinates
            .into_iter()
            .map(|coordinate| {
                canonical_rational(coordinate).ok_or(CredalContractError::InvalidRationalCoordinate)
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    /// Converts the exact values of finite binary64 coordinates.
    ///
    /// # Errors
    /// Returns an error for an empty point or nonfinite input.
    pub fn from_binary64(coordinates: &[f64]) -> Result<Self, CredalContractError> {
        if coordinates.is_empty() {
            return Err(CredalContractError::Empty);
        }
        coordinates
            .iter()
            .copied()
            .map(rational)
            .collect::<Result<_, _>>()
            .map(Self)
    }

    /// Borrows the coordinates in their declared order.
    #[must_use]
    pub fn coordinates(&self) -> &[BigRational] {
        &self.0
    }
}

/// One constraint of the form coefficients times the parameter vector <= upper.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearInequality {
    coefficients: SampleVector,
    upper: f64,
}

impl LinearInequality {
    /// Constructs a constraint using a finite coefficient row.
    ///
    /// # Errors
    /// Returns an error if the upper bound is nonfinite.
    pub fn new(coefficients: SampleVector, upper: f64) -> Result<Self, CredalContractError> {
        Ok(Self {
            coefficients,
            upper: finite(upper)?,
        })
    }

    /// Returns the coefficient row and its upper bound.
    #[must_use]
    pub fn row(&self) -> (&[f64], f64) {
        (self.coefficients.as_slice(), self.upper)
    }
}

/// A finite box intersected with an ordered list of closed linear halfspaces.
#[derive(Clone, Debug, PartialEq)]
pub struct CredalPolytope {
    parameters: Vec<Identifier>,
    bounds: Vec<CredalInterval>,
    inequalities: Vec<LinearInequality>,
}

impl CredalPolytope {
    /// Constructs support in the declared coordinate order.
    ///
    /// A box can be represented with an empty inequality list. General
    /// feasibility for a nondegenerate box is a numerical solver obligation.
    ///
    /// # Errors
    /// Returns an error for empty or repeated parameter IDs, inconsistent
    /// dimensions, or a pinned point excluded by an inequality.
    pub fn new(
        parameters: Vec<Identifier>,
        bounds: Vec<CredalInterval>,
        inequalities: Vec<LinearInequality>,
    ) -> Result<Self, CredalContractError> {
        check_identifiers(parameters.iter())?;
        check_length(parameters.len(), bounds.len())?;
        for inequality in &inequalities {
            check_length(parameters.len(), inequality.coefficients.len())?;
        }
        let polytope = Self {
            parameters,
            bounds,
            inequalities,
        };
        if let Some(point) = polytope.fixed_point() {
            if !polytope.contains(&point)? {
                return Err(CredalContractError::InfeasibleSingleton);
            }
        }
        Ok(polytope)
    }

    /// Returns the coordinate IDs and their bounds in declaration order.
    #[must_use]
    pub fn coordinates(&self) -> (&[Identifier], &[CredalInterval]) {
        (&self.parameters, &self.bounds)
    }

    /// Returns additional halfspaces in declaration order.
    #[must_use]
    pub fn inequalities(&self) -> &[LinearInequality] {
        &self.inequalities
    }

    /// Tests membership using exact comparisons of the binary64 inputs.
    ///
    /// # Errors
    /// Returns an error for a dimension mismatch or nonfinite input.
    pub fn contains(&self, point: &[f64]) -> Result<bool, CredalContractError> {
        check_length(self.parameters.len(), point.len())?;
        self.contains_exact(&CredalPoint::from_binary64(point)?)
    }

    /// Tests a rational witness against the box and halfspaces.
    ///
    /// # Errors
    /// Returns an error if the point's dimension differs from that of the support.
    pub fn contains_exact(&self, point: &CredalPoint) -> Result<bool, CredalContractError> {
        let values = point.coordinates();
        check_length(self.parameters.len(), values.len())?;
        for (bound, value) in self.bounds.iter().zip(values) {
            if value < &rational(bound.lower)? || value > &rational(bound.upper)? {
                return Ok(false);
            }
        }
        for inequality in &self.inequalities {
            if exact_dot(inequality.coefficients.as_slice(), values)? > rational(inequality.upper)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Evaluates the affine net benefits directly when every interval is pinned.
    ///
    /// Returns `None` when the declared intervals need numerical optimisation.
    ///
    /// # Errors
    /// Returns an error for empty or repeated strategies, incompatible
    /// dimensions, or a nonfinite calculation.
    pub fn point_expected_net_benefits(
        &self,
        strategies: &[AffineNetBenefit],
    ) -> Result<Option<BTreeMap<StrategyId, f64>>, CredalContractError> {
        check_identifiers(strategies.iter().map(|value| &value.strategy))?;
        for value in strategies {
            check_length(self.parameters.len(), value.coefficients.len())?;
        }
        let Some(point) = self.fixed_point() else {
            return Ok(None);
        };
        let point = CredalPoint::from_binary64(&point)?;
        let values = strategies
            .iter()
            .map(|value| {
                let benefit = value.evaluate_exact(&point)?;
                Ok((value.strategy.clone(), round_to_binary64(&benefit)?))
            })
            .collect::<Result<_, _>>()?;
        Ok(Some(values))
    }

    fn fixed_point(&self) -> Option<Vec<f64>> {
        self.bounds
            .iter()
            .map(|bound| {
                (bound.lower.partial_cmp(&bound.upper) == Some(Ordering::Equal))
                    .then_some(bound.lower)
            })
            .collect()
    }

    fn support_row_count(&self) -> usize {
        self.inequalities.len() + 2 * self.bounds.len()
    }
}

/// Expected net benefit expressed in the polytope's parameter coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineNetBenefit {
    strategy: StrategyId,
    intercept: f64,
    coefficients: SampleVector,
}

impl AffineNetBenefit {
    /// Constructs the affine objective for one strategy.
    ///
    /// # Errors
    /// Returns an error if the intercept is nonfinite.
    pub fn new(
        strategy: StrategyId,
        intercept: f64,
        coefficients: SampleVector,
    ) -> Result<Self, CredalContractError> {
        Ok(Self {
            strategy,
            intercept: finite(intercept)?,
            coefficients,
        })
    }

    /// Returns the strategy ID, intercept and coefficient row.
    #[must_use]
    pub fn objective(&self) -> (&StrategyId, f64, &[f64]) {
        (&self.strategy, self.intercept, self.coefficients.as_slice())
    }

    /// Evaluates the affine function exactly before rounding the result to binary64.
    ///
    /// # Errors
    /// Returns an error for a dimension mismatch, a nonfinite input or a
    /// result outside the finite binary64 range.
    pub fn evaluate(&self, point: &[f64]) -> Result<f64, CredalContractError> {
        check_length(self.coefficients.len(), point.len())?;
        let value = self.evaluate_exact(&CredalPoint::from_binary64(point)?)?;
        round_to_binary64(&value)
    }

    /// Returns the exact affine benefit at a rational witness.
    ///
    /// # Errors
    /// Returns an error if the point's dimension differs from the coefficient length.
    pub fn evaluate_exact(&self, point: &CredalPoint) -> Result<BigRational, CredalContractError> {
        Ok(rational(self.intercept)?
            + exact_dot(self.coefficients.as_slice(), point.coordinates())?)
    }
}

/// Farkas multipliers and the stated positive margin for a dominated strategy.
///
/// Numerical verification must bind these values to the objectives and support.
/// The margin is the advantage of the alternative mixture after normalising
/// its multipliers to sum to one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DominanceWitness {
    strategy: StrategyId,
    alternative_multipliers: BTreeMap<StrategyId, BigRational>,
    support_multipliers: Vec<BigRational>,
    margin: BigRational,
}

impl DominanceWitness {
    /// Constructs a certificate from the exact values of binary64 inputs.
    ///
    /// Support multipliers use declared halfspaces first, then two rows
    /// for each coordinate in order. For coordinate x, those rows are
    /// -x <= -lower and x <= upper.
    /// An alternative multiplier refers to the row U(alternative) - U(strategy) <= 0.
    ///
    /// # Errors
    /// Returns an error for inconsistent support dimensions, nonfinite or
    /// negative multipliers, or a zero combination of alternative strategies.
    pub fn new(
        polytope: &CredalPolytope,
        strategy: StrategyId,
        alternative_multipliers: BTreeMap<StrategyId, f64>,
        support_multipliers: Vec<f64>,
        margin: Threshold,
    ) -> Result<Self, CredalContractError> {
        check_length(polytope.support_row_count(), support_multipliers.len())?;
        let weight =
            |value| BigRational::from_float(value).ok_or(CredalContractError::InvalidMultipliers);
        let alternatives = alternative_multipliers
            .into_iter()
            .map(|(id, value)| Ok((id, weight(value)?)))
            .collect::<Result<_, _>>()?;
        let support = support_multipliers
            .into_iter()
            .map(weight)
            .collect::<Result<_, _>>()?;
        Self::new_exact(
            polytope,
            strategy,
            alternatives,
            support,
            rational(margin.get())?,
        )
    }

    /// Constructs an exact certificate, including non-binary rational margins or weights.
    /// Support rows follow the order documented by [`Self::new`].
    ///
    /// # Errors
    /// Returns an error for inconsistent support dimensions, malformed or negative
    /// weights, a zero alternative combination, or a malformed or nonpositive margin.
    pub fn new_exact(
        polytope: &CredalPolytope,
        strategy: StrategyId,
        alternative_multipliers: BTreeMap<StrategyId, BigRational>,
        support_multipliers: Vec<BigRational>,
        margin: BigRational,
    ) -> Result<Self, CredalContractError> {
        check_length(polytope.support_row_count(), support_multipliers.len())?;
        let weight = |value| {
            canonical_rational(value)
                .filter(|weight| !weight.is_negative())
                .ok_or(CredalContractError::InvalidMultipliers)
        };
        let alternative_multipliers: BTreeMap<_, _> = alternative_multipliers
            .into_iter()
            .map(|(id, value)| Ok((id, weight(value)?)))
            .collect::<Result<_, _>>()?;
        let support_multipliers = support_multipliers
            .into_iter()
            .map(weight)
            .collect::<Result<_, _>>()?;
        if alternative_multipliers.contains_key(&strategy)
            || !alternative_multipliers.values().any(Signed::is_positive)
        {
            return Err(CredalContractError::InvalidMultipliers);
        }
        let margin = canonical_rational(margin)
            .filter(Signed::is_positive)
            .ok_or(CredalContractError::InvalidMargin)?;
        Ok(Self {
            strategy,
            alternative_multipliers,
            support_multipliers,
            margin,
        })
    }

    /// Returns the strategy that the certificate seeks to exclude.
    #[must_use]
    pub fn strategy(&self) -> &StrategyId {
        &self.strategy
    }

    /// Borrows the exact alternative multipliers, support multipliers and margin.
    #[must_use]
    pub fn certificate(
        &self,
    ) -> (
        &BTreeMap<StrategyId, BigRational>,
        &[BigRational],
        &BigRational,
    ) {
        (
            &self.alternative_multipliers,
            &self.support_multipliers,
            &self.margin,
        )
    }
}

/// Result cases for E-admissibility verification.
#[derive(Clone, Debug, PartialEq)]
pub enum EAdmissibilityResult {
    /// A feasible point where the strategy attains the maximum expected benefit.
    Admissible {
        /// The strategy whose optimality is witnessed.
        strategy: StrategyId,
        /// The parameter vector in the polytope's coordinate order.
        point: CredalPoint,
    },
    /// A strict dominance certificate excludes the strategy.
    Dominated(DominanceWitness),
    /// The solver produced no verified classification.
    Indeterminate {
        /// The strategy that remains unclassified.
        strategy: StrategyId,
    },
}

fn check_identifiers<T: Ord>(
    identifiers: impl Iterator<Item = T>,
) -> Result<(), CredalContractError> {
    let mut seen = BTreeSet::new();
    for id in identifiers {
        if !seen.insert(id) {
            return Err(CredalContractError::DuplicateIdentifier);
        }
    }
    if seen.is_empty() {
        return Err(CredalContractError::Empty);
    }
    Ok(())
}

fn check_length(expected: usize, actual: usize) -> Result<(), CredalContractError> {
    if expected != actual {
        return Err(CredalContractError::DimensionMismatch { expected, actual });
    }
    Ok(())
}

fn finite(value: f64) -> Result<f64, CredalContractError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CredalContractError::NonFinite)
    }
}

fn rational(value: f64) -> Result<BigRational, CredalContractError> {
    BigRational::from_float(value).ok_or(CredalContractError::NonFinite)
}

fn canonical_rational(value: BigRational) -> Option<BigRational> {
    let (numerator, denominator) = value.into_raw();
    (!denominator.is_zero()).then(|| BigRational::new(numerator, denominator))
}

fn round_to_binary64(value: &BigRational) -> Result<f64, CredalContractError> {
    finite(value.to_f64().ok_or(CredalContractError::NonFinite)?)
}

fn exact_dot(
    coefficients: &[f64],
    point: &[BigRational],
) -> Result<BigRational, CredalContractError> {
    check_length(coefficients.len(), point.len())?;
    coefficients
        .iter()
        .zip(point)
        .try_fold(BigRational::zero(), |sum, (&coefficient, coordinate)| {
            Ok(sum + rational(coefficient)? * coordinate)
        })
}
