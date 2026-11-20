use std::cmp::Ordering;
use std::collections::{btree_map::Entry, BTreeMap};
use std::num::NonZeroUsize;

use voiage_domain::{
    AffineNetBenefit, BigRational, CredalPoint, CredalPolytope, DominanceWitness,
    EAdmissibilityResult, StrategyId,
};

use super::{
    exact, index_strategies, input_error, verify_e_admissibility, verify_strict_credal_dominance,
};
use crate::NumericalInputError;

/// An exact affine net benefit and a point attaining it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredalExtremum {
    /// Exact value of the objective at the witness.
    pub value: BigRational,
    /// Feasible parameter coordinates in declaration order.
    pub point: CredalPoint,
}

/// Attained bounds of one strategy's affine net benefit over the credal set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredalBenefitRange {
    /// Minimum benefit over the support.
    pub minimum: CredalExtremum,
    /// Maximum benefit over the support.
    pub maximum: CredalExtremum,
}

/// Decision rules evaluated on nonempty support, in strategy ID order.
#[derive(Clone, Debug, PartialEq)]
pub struct CredalAnalysis {
    /// Exact bounds for each strategy.
    pub ranges: BTreeMap<StrategyId, CredalBenefitRange>,
    /// Strategies with the highest minimum benefit, including ties.
    pub gamma_maximin: Vec<StrategyId>,
    /// Strategies with the highest maximum benefit, including ties.
    pub gamma_maximax: Vec<StrategyId>,
    /// Strategies whose maximum is at least every other strategy's minimum.
    pub interval_undominated: Vec<StrategyId>,
    /// Verified witnesses or an indeterminate outcome for each strategy.
    pub e_admissibility: Vec<EAdmissibilityResult>,
    /// Number of constraint bases examined under the shared budget.
    pub bases_examined: usize,
}

/// Result of evaluating affine benefits over a bounded credal set.
#[derive(Clone, Debug, PartialEq)]
pub enum CredalSolveResult {
    /// The support bounds and range-based decision rules are certified.
    Feasible(CredalAnalysis),
    /// Exhaustive enumeration proved the support empty.
    Infeasible {
        /// Number of constraint bases examined.
        bases_examined: usize,
    },
    /// The budget expired before all support extrema could be certified.
    Indeterminate {
        /// Number of constraint bases examined.
        bases_examined: usize,
    },
}

/// Solves the range, E-admissibility and dominance programs by exact basis enumeration.
///
/// Inputs retain the exact values of their binary64 coordinates and coefficients.
/// The basis budget is shared across the support and all strategy programs. If
/// exhausted after the range calculation, only the unresolved strategies are
/// indeterminate. A fully pinned box needs no basis search. Enumeration is
/// combinatorial, so callers choose the budget for their problem size.
///
/// # Errors
/// Returns an error for empty or duplicate strategies or incompatible dimensions.
pub fn solve_credal(
    support: &CredalPolytope,
    strategies: &[AffineNetBenefit],
    max_bases: NonZeroUsize,
) -> Result<CredalSolveResult, NumericalInputError> {
    let problem = DecisionProblem::new(support, strategies)?;
    let mut examined = 0;
    let mut ranges = BTreeMap::new();
    let (_, bounds) = support.coordinates();
    let pinned: Option<Vec<_>> = bounds
        .iter()
        .map(|bound| {
            let (lo, hi) = bound.endpoints();
            (lo.partial_cmp(&hi) == Some(Ordering::Equal)).then_some(lo)
        })
        .collect();
    if let Some(point) = &pinned {
        let point =
            CredalPoint::from_binary64(point).map_err(|error| input_error("point", error))?;
        record_extrema(&problem.objectives, &point, &mut ranges);
    } else if matches!(
        visit_feasible_bases(
            &problem.support_rows,
            bounds.len(),
            max_bases,
            &mut examined,
            |_, point| {
                let point = CredalPoint::new(point).map_err(|error| input_error("point", error))?;
                record_extrema(&problem.objectives, &point, &mut ranges);
                Ok(None::<()>)
            }
        )?,
        BasisOutcome::BudgetExhausted,
    ) {
        return Ok(CredalSolveResult::Indeterminate {
            bases_examined: examined,
        });
    }

    let (Some(best_lower), Some(best_upper)) = (
        ranges.values().map(|range| &range.minimum.value).max(),
        ranges.values().map(|range| &range.maximum.value).max(),
    ) else {
        return Ok(CredalSolveResult::Infeasible {
            bases_examined: examined,
        });
    };
    let mut gamma_maximin = Vec::new();
    let mut gamma_maximax = Vec::new();
    let mut interval_undominated = Vec::new();
    for (id, range) in &ranges {
        if &range.minimum.value == best_lower {
            gamma_maximin.push(id.clone());
        }
        if &range.maximum.value == best_upper {
            gamma_maximax.push(id.clone());
        }
        if &range.maximum.value >= best_lower {
            interval_undominated.push(id.clone());
        }
    }

    let mut e_admissibility = Vec::with_capacity(ranges.len());
    let fixed_best = pinned.and_then(|_| gamma_maximin.first());
    for target in &problem.objectives {
        let maximum = &ranges[target.id].maximum;
        if verify_e_admissibility(support, strategies, target.id, &maximum.point)? {
            e_admissibility.push(EAdmissibilityResult::Admissible {
                strategy: target.id.clone(),
                point: maximum.point.clone(),
            });
            continue;
        }
        let witness = if let Some(best) = fixed_best {
            problem.dominance(
                target.id,
                &maximum.point,
                BTreeMap::from([(best.clone(), BigRational::from_integer(1.into()))]),
                vec![BigRational::default(); problem.support_rows.len()],
                best_lower - &maximum.value,
            )?
        } else {
            problem.search_strategy(target, &maximum.point, max_bases, &mut examined)?
        };
        e_admissibility.push(
            witness.unwrap_or_else(|| EAdmissibilityResult::Indeterminate {
                strategy: target.id.clone(),
            }),
        );
    }
    Ok(CredalSolveResult::Feasible(CredalAnalysis {
        ranges,
        gamma_maximin,
        gamma_maximax,
        interval_undominated,
        e_admissibility,
        bases_examined: examined,
    }))
}

#[derive(Clone)]
struct Row {
    coefficients: Vec<BigRational>,
    upper: BigRational,
}

struct Objective<'a> {
    id: &'a StrategyId,
    coefficients: Vec<BigRational>,
    intercept: BigRational,
}

struct DecisionProblem<'a> {
    support: &'a CredalPolytope,
    strategies: &'a [AffineNetBenefit],
    support_rows: Vec<Row>,
    objectives: Vec<Objective<'a>>,
}

impl<'a> DecisionProblem<'a> {
    fn new(
        support: &'a CredalPolytope,
        strategies: &'a [AffineNetBenefit],
    ) -> Result<Self, NumericalInputError> {
        let objectives = index_strategies(support, strategies)?
            .into_iter()
            .map(|(id, benefit)| {
                let (_, intercept, coefficients) = benefit.objective();
                Ok(Objective {
                    id,
                    intercept: exact(intercept)?,
                    coefficients: coefficients
                        .iter()
                        .copied()
                        .map(exact)
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<_, NumericalInputError>>()?;
        let (_, bounds) = support.coordinates();
        let mut support_rows = Vec::with_capacity(support.inequalities().len() + 2 * bounds.len());
        for inequality in support.inequalities() {
            let (coefficients, upper) = inequality.row();
            support_rows.push(Row {
                coefficients: coefficients
                    .iter()
                    .copied()
                    .map(exact)
                    .collect::<Result<_, _>>()?,
                upper: exact(upper)?,
            });
        }
        for (coordinate, bound) in bounds.iter().enumerate() {
            let (lo, hi) = bound.endpoints();
            let mut lower = vec![BigRational::default(); bounds.len()];
            let mut upper = lower.clone();
            lower[coordinate] = BigRational::from_integer((-1).into());
            upper[coordinate] = BigRational::from_integer(1.into());
            support_rows.push(Row {
                coefficients: lower,
                upper: -exact(lo)?,
            });
            support_rows.push(Row {
                coefficients: upper,
                upper: exact(hi)?,
            });
        }
        Ok(Self {
            support,
            strategies,
            support_rows,
            objectives,
        })
    }

    fn search_strategy(
        &self,
        target: &Objective<'_>,
        feasible_point: &CredalPoint,
        max_bases: NonZeroUsize,
        examined: &mut usize,
    ) -> Result<Option<EAdmissibilityResult>, NumericalInputError> {
        let zero = BigRational::default();
        let minus_one = BigRational::from_integer((-1).into());
        let mut rows = self.support_rows.clone();
        for row in &mut rows {
            row.coefficients.push(zero.clone());
        }
        let alternatives: Vec<_> = self
            .objectives
            .iter()
            .filter(|other| other.id != target.id)
            .collect();
        // Minimize t subject to U(other) - U(target) <= t and the support rows.
        // Bounded parameter coordinates make the epigraph pointed. If there
        // is an alternative, an optimum is attained at a feasible basis.
        for other in &alternatives {
            let mut coefficients: Vec<_> = other
                .coefficients
                .iter()
                .zip(&target.coefficients)
                .map(|(other, target)| other - target)
                .collect();
            coefficients.push(minus_one.clone());
            rows.push(Row {
                coefficients,
                upper: &target.intercept - &other.intercept,
            });
        }
        let dimension = target.coefficients.len() + 1;
        let mut stationarity_rhs = vec![zero.clone(); dimension];
        stationarity_rhs[dimension - 1] = minus_one;
        let outcome =
            visit_feasible_bases(&rows, dimension, max_bases, examined, |basis, mut point| {
                let Some(margin) = point.pop() else {
                    return Ok(None);
                };
                if margin <= zero {
                    let point =
                        CredalPoint::new(point).map_err(|error| input_error("point", error))?;
                    let checked =
                        verify_e_admissibility(self.support, self.strategies, target.id, &point)?;
                    return Ok(checked.then(|| EAdmissibilityResult::Admissible {
                        strategy: target.id.clone(),
                        point,
                    }));
                }
                // Solve A_basis^T * weights = -e_t. Nonnegative weights give a
                // dual bound, and the t coordinate makes alternative weights sum to one.
                let transposed = (0..dimension)
                    .map(|column| {
                        basis
                            .iter()
                            .map(|&index| rows[index].coefficients[column].clone())
                            .collect()
                    })
                    .collect();
                let Some(weights) = solve_square(transposed, stationarity_rhs.clone()) else {
                    return Ok(None);
                };
                if weights.iter().any(|weight| weight < &zero) {
                    return Ok(None);
                }
                let mut support_weights = vec![zero.clone(); self.support_rows.len()];
                let mut alternative_weights = BTreeMap::new();
                for (&row, weight) in basis
                    .iter()
                    .zip(weights)
                    .filter(|(_, weight)| weight > &zero)
                {
                    if row < support_weights.len() {
                        support_weights[row] = weight;
                    } else {
                        alternative_weights
                            .insert(alternatives[row - support_weights.len()].id.clone(), weight);
                    }
                }
                self.dominance(
                    target.id,
                    feasible_point,
                    alternative_weights,
                    support_weights,
                    margin,
                )
            })?;
        match outcome {
            BasisOutcome::Found(witness) => Ok(Some(witness)),
            BasisOutcome::Complete | BasisOutcome::BudgetExhausted => Ok(None),
        }
    }

    fn dominance(
        &self,
        strategy: &StrategyId,
        feasible_point: &CredalPoint,
        alternatives: BTreeMap<StrategyId, BigRational>,
        support: Vec<BigRational>,
        margin: BigRational,
    ) -> Result<Option<EAdmissibilityResult>, NumericalInputError> {
        let witness = DominanceWitness::new_exact(
            self.support,
            strategy.clone(),
            alternatives,
            support,
            margin,
        )
        .map_err(|error| input_error("certificate", error))?;
        let checked = verify_strict_credal_dominance(
            self.support,
            self.strategies,
            feasible_point,
            &witness,
        )?;
        Ok(checked.then_some(EAdmissibilityResult::Dominated(witness)))
    }
}

fn record_extrema(
    objectives: &[Objective<'_>],
    point: &CredalPoint,
    ranges: &mut BTreeMap<StrategyId, CredalBenefitRange>,
) {
    for objective in objectives {
        let value = &objective.intercept + dot(&objective.coefficients, point.coordinates());
        let extremum = || CredalExtremum {
            value: value.clone(),
            point: point.clone(),
        };
        match ranges.entry(objective.id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(CredalBenefitRange {
                    minimum: extremum(),
                    maximum: extremum(),
                });
            }
            Entry::Occupied(mut entry) => {
                let range = entry.get_mut();
                if value < range.minimum.value {
                    range.minimum = extremum();
                }
                if value > range.maximum.value {
                    range.maximum = extremum();
                }
            }
        }
    }
}

enum BasisOutcome<T> {
    Complete,
    BudgetExhausted,
    Found(T),
}

fn visit_feasible_bases<T>(
    rows: &[Row],
    dimension: usize,
    max_bases: NonZeroUsize,
    examined: &mut usize,
    mut visit: impl FnMut(&[usize], Vec<BigRational>) -> Result<Option<T>, NumericalInputError>,
) -> Result<BasisOutcome<T>, NumericalInputError> {
    let Some(last_start) = rows.len().checked_sub(dimension) else {
        return Ok(BasisOutcome::Complete);
    };
    let mut basis: Vec<_> = (0..dimension).collect();
    loop {
        if *examined == max_bases.get() {
            return Ok(BasisOutcome::BudgetExhausted);
        }
        *examined += 1;
        let matrix = basis
            .iter()
            .map(|&index| rows[index].coefficients.clone())
            .collect();
        let upper = basis
            .iter()
            .map(|&index| rows[index].upper.clone())
            .collect();
        if let Some(point) = solve_square(matrix, upper) {
            if rows
                .iter()
                .all(|row| dot(&row.coefficients, &point) <= row.upper)
            {
                if let Some(witness) = visit(&basis, point)? {
                    return Ok(BasisOutcome::Found(witness));
                }
            }
        }
        let Some(next) = (0..dimension).rfind(|&index| basis[index] < last_start + index) else {
            return Ok(BasisOutcome::Complete);
        };
        let start = basis[next] + 1;
        for (offset, index) in basis[next..].iter_mut().enumerate() {
            *index = start + offset;
        }
    }
}

fn solve_square(
    mut matrix: Vec<Vec<BigRational>>,
    mut rhs: Vec<BigRational>,
) -> Option<Vec<BigRational>> {
    let zero = BigRational::default();
    for column in 0..matrix.len() {
        let pivot = (column..matrix.len()).find(|&row| matrix[row][column] != zero)?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let (solved, pending) = matrix.split_at_mut(column + 1);
        let (known_rhs, pending_rhs) = rhs.split_at_mut(column + 1);
        let pivot_row = &solved[column];
        for (row, value) in pending.iter_mut().zip(pending_rhs) {
            if row[column] == zero {
                continue;
            }
            let factor = &row[column] / &pivot_row[column];
            for (coefficient, pivot_value) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *coefficient -= &factor * pivot_value;
            }
            *value -= &factor * &known_rhs[column];
        }
    }
    for (index, row) in matrix.iter().enumerate().rev() {
        let known = dot(&row[index + 1..], &rhs[index + 1..]);
        rhs[index] = (&rhs[index] - known) / &row[index];
    }
    Some(rhs)
}

fn dot(lhs: &[BigRational], rhs: &[BigRational]) -> BigRational {
    lhs.iter().zip(rhs).map(|(left, right)| left * right).sum()
}
