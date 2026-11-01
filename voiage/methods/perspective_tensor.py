# Adapted from edithatogo/vop_poc_nz at ad84d3090a2d97f2805a86e94a7b9614a3d1ab14 (Apache-2.0).
"""Decision-theoretic Value of Perspective reference implementation.

This module treats strategies as the alternatives and perspectives as evaluative
lenses. The primary quantity is current-information directional expected
perspective regret. Per-draw regret is exposed only as a diagnostic. The module
also provides tie-aware perspective acceptability probabilities and an exact
expected-value frontier over convex mixtures of two perspectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
lazy from numpy.typing import NDArray

from voiage.exceptions import InputError
from voiage.methods.perspective import METHOD_CONTRACT_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class PerspectiveError(InputError):
    """Invalid perspective-analysis input."""


class DecisionRule(StrEnum):
    """Supported decision rules."""

    EXPECTED_VALUE = "expected_value"
    PER_DRAW = "per_draw"


class TiePolicy(StrEnum):
    """How exact or numerical ties are handled."""

    FIRST = "first"
    SPLIT = "split"
    ERROR = "error"


@dataclass(frozen=True)
class PerspectiveRegret:
    """Directional loss from choosing under one lens and evaluating under another."""

    choose_under: str
    evaluate_under: str
    decision_rule: DecisionRule
    chosen_strategy: str
    target_strategy: str
    per_person: float
    population: float | None = None
    discordance_probability: float | None = None
    selection_tie_policy: TiePolicy = TiePolicy.FIRST
    tie_detected: bool = False
    method_contract_version: str = METHOD_CONTRACT_VERSION
    per_draw_loss: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def population_value(self) -> float | None:
        """Scale per-person regret to the specified population."""
        if self.population is None:
            return None
        return self.per_person * self.population

    def as_dict(self, *, include_distribution: bool = False) -> dict[str, Any]:
        """Serialize the result using the perspective contract field names."""
        output: dict[str, Any] = {
            "choose_under": self.choose_under,
            "evaluate_under": self.evaluate_under,
            "decision_rule": self.decision_rule.value,
            "chosen_strategy": self.chosen_strategy,
            "target_strategy": self.target_strategy,
            "per_person": self.per_person,
            "population": self.population,
            "population_value": self.population_value,
            "discordance_probability": self.discordance_probability,
            "selection_tie_policy": self.selection_tie_policy.value,
            "tie_detected": self.tie_detected,
            "method_contract_version": self.method_contract_version,
        }
        if include_distribution and self.per_draw_loss is not None:
            output["per_draw_loss"] = self.per_draw_loss.tolist()
        return output


@dataclass(frozen=True)
class PerspectiveAcceptabilityRow:
    """Probability of optimality and expected-value rank for one strategy."""

    perspective: str
    strategy: str
    probability_optimal: float
    expected_net_benefit: float
    expected_value_rank: int
    tie_policy: TiePolicy = TiePolicy.SPLIT
    method_contract_version: str = METHOD_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result using the perspective contract field names."""
        return {
            "perspective": self.perspective,
            "strategy": self.strategy,
            "probability_optimal": self.probability_optimal,
            "expected_net_benefit": self.expected_net_benefit,
            "expected_value_rank": self.expected_value_rank,
            "tie_policy": self.tie_policy.value,
            "method_contract_version": self.method_contract_version,
        }


@dataclass(frozen=True)
class PerspectiveMixtureFrontierRow:
    """Strategy statistics at a sampled mixture weight."""

    left_perspective: str
    right_perspective: str
    right_weight: float
    strategy: str
    probability_optimal: float
    expected_net_benefit: float
    expected_value_rank: int
    tie_policy: TiePolicy = TiePolicy.SPLIT
    method_contract_version: str = METHOD_CONTRACT_VERSION

    @property
    def left_weight(self) -> float:
        """Return the complement of the right-perspective weight."""
        return 1.0 - self.right_weight

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result using the perspective contract field names."""
        return {
            "left_perspective": self.left_perspective,
            "right_perspective": self.right_perspective,
            "left_weight": self.left_weight,
            "right_weight": self.right_weight,
            "strategy": self.strategy,
            "probability_optimal": self.probability_optimal,
            "expected_net_benefit": self.expected_net_benefit,
            "expected_value_rank": self.expected_value_rank,
            "tie_policy": self.tie_policy.value,
            "method_contract_version": self.method_contract_version,
        }


@dataclass(frozen=True)
class PerspectiveFrontierSegment:
    """An exact interval of perspective weights with the same expected-value optimum."""

    left_perspective: str
    right_perspective: str
    lower_right_weight: float
    upper_right_weight: float
    optimal_strategies: tuple[str, ...]
    midpoint_expected_net_benefit: float
    method_contract_version: str = METHOD_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result using the perspective contract field names."""
        return {
            "left_perspective": self.left_perspective,
            "right_perspective": self.right_perspective,
            "lower_right_weight": self.lower_right_weight,
            "upper_right_weight": self.upper_right_weight,
            "optimal_strategies": list(self.optimal_strategies),
            "midpoint_expected_net_benefit": self.midpoint_expected_net_benefit,
            "method_contract_version": self.method_contract_version,
        }


@dataclass(frozen=True)
class PerspectiveSwitchPoint:
    """Change of optimal strategies along a two-perspective mixture."""

    left_perspective: str
    right_perspective: str
    right_weight: float
    from_strategies: tuple[str, ...]
    to_strategies: tuple[str, ...]
    method_contract_version: str = METHOD_CONTRACT_VERSION

    @property
    def from_strategy(self) -> str:
        """Backward-compatible scalar name when a single strategy is optimal."""
        return (
            self.from_strategies[0]
            if len(self.from_strategies) == 1
            else "|".join(self.from_strategies)
        )

    @property
    def to_strategy(self) -> str:
        """Backward-compatible scalar name when a single strategy is optimal."""
        return (
            self.to_strategies[0]
            if len(self.to_strategies) == 1
            else "|".join(self.to_strategies)
        )

    @property
    def lower_right_weight(self) -> float:
        """Return the switch weight under the interval-compatible field name."""
        return self.right_weight

    @property
    def upper_right_weight(self) -> float:
        """Return the switch weight under the interval-compatible field name."""
        return self.right_weight

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result using the perspective contract field names."""
        return {
            "left_perspective": self.left_perspective,
            "right_perspective": self.right_perspective,
            "right_weight": self.right_weight,
            "from_strategies": list(self.from_strategies),
            "to_strategies": list(self.to_strategies),
            "method_contract_version": self.method_contract_version,
        }


def _validate_population(population: float | None) -> None:
    if population is None:
        return
    value = float(population)
    if not np.isfinite(value) or value < 0:
        raise PerspectiveError("population must be finite and non-negative")


def _tie_mask(values: NDArray[np.float64], *, atol: float) -> NDArray[np.bool_]:
    maxima = np.max(values, axis=-1, keepdims=True)
    return np.isclose(values, maxima, rtol=0.0, atol=atol)


def _dense_descending_ranks(
    values: NDArray[np.float64], *, atol: float
) -> NDArray[np.int64]:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.int64)
    rank = 1
    previous: float | None = None
    for position, index in enumerate(order):
        value = float(values[index])
        if previous is not None and not np.isclose(
            value, previous, rtol=0.0, atol=atol
        ):
            rank = position + 1
        ranks[index] = rank
        previous = value
    return ranks


def _select_single(
    values: NDArray[np.float64], tie_policy: TiePolicy, *, atol: float
) -> tuple[int, bool]:
    mask = _tie_mask(values[np.newaxis, :], atol=atol)[0]
    indices = np.flatnonzero(mask)
    tied = len(indices) > 1
    if tied and tie_policy is TiePolicy.ERROR:
        raise PerspectiveError("Optimal strategy is tied under tie_policy='error'")
    return int(indices[0]), tied


@dataclass(frozen=True)
class NetBenefitTensor:
    """Dense draw x strategy x perspective net-benefit tensor."""

    values: NDArray[np.float64]
    strategies: Sequence[str]
    perspectives: Sequence[str]
    case_id: str | None = None
    draw_ids: Sequence[str] | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dimensions, labels and finite values before evaluation."""
        values = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "strategies", tuple(self.strategies))
        object.__setattr__(self, "perspectives", tuple(self.perspectives))
        if self.draw_ids is not None:
            object.__setattr__(self, "draw_ids", tuple(self.draw_ids))
        object.__setattr__(self, "attrs", dict(self.attrs))
        if values.ndim != 3:
            raise PerspectiveError(
                "values must have shape draw x strategy x perspective"
            )
        n_draws, n_strategies, n_perspectives = values.shape
        if n_draws < 1 or n_strategies < 1 or n_perspectives < 1:
            raise PerspectiveError(
                "tensor requires at least one draw, strategy, and perspective"
            )
        if n_strategies != len(self.strategies):
            raise PerspectiveError("strategy-name count does not match tensor shape")
        if n_perspectives != len(self.perspectives):
            raise PerspectiveError("perspective-name count does not match tensor shape")
        if len(set(self.strategies)) != len(self.strategies):
            raise PerspectiveError("strategy names must be unique")
        if len(set(self.perspectives)) != len(self.perspectives):
            raise PerspectiveError("perspective names must be unique")
        if self.draw_ids is not None and len(self.draw_ids) != n_draws:
            raise PerspectiveError("draw ID count does not match tensor shape")
        if not np.all(np.isfinite(values)):
            raise PerspectiveError("values must be finite")

    @property
    def n_draws(self) -> int:
        """Count the paired uncertainty draws."""
        return int(self.values.shape[0])

    @property
    def n_strategies(self) -> int:
        """Count the decision alternatives."""
        return int(self.values.shape[1])

    @property
    def n_perspectives(self) -> int:
        """Count the evaluation lenses."""
        return int(self.values.shape[2])

    def strategy_index(self, strategy: str) -> int:
        """Resolve a strategy label to its tensor position."""
        try:
            return self.strategies.index(strategy)
        except ValueError as exc:
            raise PerspectiveError(f"Unknown strategy: {strategy!r}") from exc

    def perspective_index(self, perspective: str) -> int:
        """Resolve a perspective label to its tensor position."""
        try:
            return self.perspectives.index(perspective)
        except ValueError as exc:
            raise PerspectiveError(f"Unknown perspective: {perspective!r}") from exc

    def nmb(self, perspective: str) -> NDArray[np.float64]:
        """Return the draw-by-strategy benefits for one lens."""
        return self.values[:, :, self.perspective_index(perspective)]

    def expected_net_benefit(self, perspective: str) -> NDArray[np.float64]:
        """Average each strategy's net benefit over draws."""
        return np.mean(self.nmb(perspective), axis=0)

    def optimal_strategy_index(
        self,
        perspective: str,
        *,
        decision_rule: DecisionRule | str = DecisionRule.EXPECTED_VALUE,
        tie_policy: TiePolicy | str = TiePolicy.FIRST,
        atol: float = 1e-12,
    ) -> int | NDArray[np.int64]:
        """Select an optimal strategy under the requested decision and tie rules."""
        rule = DecisionRule(decision_rule)
        policy = TiePolicy(tie_policy)
        nmb = self.nmb(perspective)
        if rule is DecisionRule.EXPECTED_VALUE:
            index, _ = _select_single(np.mean(nmb, axis=0), policy, atol=atol)
            return index
        mask = _tie_mask(nmb, atol=atol)
        if policy is TiePolicy.ERROR and np.any(np.sum(mask, axis=1) > 1):
            raise PerspectiveError("At least one PSA draw has a tied optimum")
        return np.argmax(mask, axis=1).astype(np.int64)

    def discordance_probability(
        self,
        perspective_a: str,
        perspective_b: str,
        *,
        tie_policy: TiePolicy | str = TiePolicy.SPLIT,
        atol: float = 1e-12,
    ) -> float:
        """Estimate the probability of different optimal choices on paired draws."""
        policy = TiePolicy(tie_policy)
        a_mask = _tie_mask(self.nmb(perspective_a), atol=atol)
        b_mask = _tie_mask(self.nmb(perspective_b), atol=atol)
        if policy is TiePolicy.ERROR and (
            np.any(np.sum(a_mask, axis=1) > 1) or np.any(np.sum(b_mask, axis=1) > 1)
        ):
            raise PerspectiveError("Tied draw-level optima detected")
        if policy is TiePolicy.FIRST:
            return float(
                np.mean(np.argmax(a_mask, axis=1) != np.argmax(b_mask, axis=1))
            )
        a_count = np.sum(a_mask, axis=1)
        b_count = np.sum(b_mask, axis=1)
        overlap = np.sum(a_mask & b_mask, axis=1)
        probability_same = overlap / (a_count * b_count)
        return float(np.mean(1.0 - probability_same))

    def evop(
        self,
        *,
        choose_under: str,
        evaluate_under: str,
        decision_rule: DecisionRule | str = DecisionRule.EXPECTED_VALUE,
        population: float | None = None,
        keep_distribution: bool = False,
        selection_tie_policy: TiePolicy | str = TiePolicy.FIRST,
        atol: float = 1e-12,
    ) -> PerspectiveRegret:
        """Evaluate directional regret with optional population scaling."""
        _validate_population(population)
        rule = DecisionRule(decision_rule)
        tie_policy = TiePolicy(selection_tie_policy)
        eval_nmb = self.nmb(evaluate_under)
        tie_detected = False
        if rule is DecisionRule.EXPECTED_VALUE:
            choose_values = self.expected_net_benefit(choose_under)
            target_values = self.expected_net_benefit(evaluate_under)
            chosen_mask = _tie_mask(choose_values[np.newaxis, :], atol=atol)[0]
            target_mask = _tie_mask(target_values[np.newaxis, :], atol=atol)[0]
            chosen_tie = bool(np.sum(chosen_mask) > 1)
            target_tie = bool(np.sum(target_mask) > 1)
            tie_detected = chosen_tie or target_tie
            if tie_policy is TiePolicy.ERROR and tie_detected:
                raise PerspectiveError("Tied expected-value optima detected")
            if tie_policy is TiePolicy.SPLIT:
                chosen_values = np.mean(eval_nmb[:, chosen_mask], axis=1)
                target_selected_values = np.mean(eval_nmb[:, target_mask], axis=1)
                chosen_name = "|".join(
                    np.asarray(self.strategies, dtype=object)[chosen_mask].tolist()
                )
                target_name = "|".join(
                    np.asarray(self.strategies, dtype=object)[target_mask].tolist()
                )
            else:
                chosen_idx = int(np.flatnonzero(chosen_mask)[0])
                target_idx = int(np.flatnonzero(target_mask)[0])
                chosen_values = eval_nmb[:, chosen_idx]
                target_selected_values = eval_nmb[:, target_idx]
                chosen_name = self.strategies[chosen_idx]
                target_name = self.strategies[target_idx]
            loss = target_selected_values - chosen_values
            per_person = max(float(np.mean(loss)), 0.0)
        else:
            chosen_mask = _tie_mask(self.nmb(choose_under), atol=atol)
            target_mask = _tie_mask(self.nmb(evaluate_under), atol=atol)
            tie_detected = bool(
                np.any(np.sum(chosen_mask, axis=1) > 1)
                or np.any(np.sum(target_mask, axis=1) > 1)
            )
            if tie_policy is TiePolicy.ERROR and tie_detected:
                raise PerspectiveError("Tied draw-level optima detected")
            if tie_policy is TiePolicy.SPLIT:
                chosen_values = np.sum(eval_nmb * chosen_mask, axis=1) / np.sum(
                    chosen_mask, axis=1
                )
                target_selected_values = np.sum(
                    eval_nmb * target_mask, axis=1
                ) / np.sum(target_mask, axis=1)
            else:
                chosen_idx = np.argmax(chosen_mask, axis=1)
                target_idx = np.argmax(target_mask, axis=1)
                draw_index = np.arange(self.n_draws)
                chosen_values = eval_nmb[draw_index, chosen_idx]
                target_selected_values = eval_nmb[draw_index, target_idx]
            loss = np.maximum(target_selected_values - chosen_values, 0.0)
            per_person = float(np.mean(loss))
            chosen_name = "<per_draw>"
            target_name = "<per_draw>"
        return PerspectiveRegret(
            choose_under=choose_under,
            evaluate_under=evaluate_under,
            decision_rule=rule,
            chosen_strategy=chosen_name,
            target_strategy=target_name,
            per_person=per_person,
            population=population,
            discordance_probability=self.discordance_probability(
                choose_under, evaluate_under, tie_policy=TiePolicy.SPLIT, atol=atol
            ),
            selection_tie_policy=tie_policy,
            tie_detected=tie_detected,
            per_draw_loss=loss if keep_distribution else None,
        )

    def regret_matrix(
        self,
        *,
        decision_rule: DecisionRule | str = DecisionRule.EXPECTED_VALUE,
        population: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return directional regret records for ordered perspective pairs."""
        return [
            self.evop(
                choose_under=source,
                evaluate_under=target,
                decision_rule=decision_rule,
                population=population,
            ).as_dict()
            for source in self.perspectives
            for target in self.perspectives
        ]

    def perspective_acceptability_frontier(
        self,
        *,
        tie_policy: TiePolicy | str = TiePolicy.SPLIT,
        atol: float = 1e-12,
    ) -> list[PerspectiveAcceptabilityRow]:
        """Compute draw-level optimality probabilities and expected-value ranks."""
        policy = TiePolicy(tie_policy)
        rows: list[PerspectiveAcceptabilityRow] = []
        for perspective in self.perspectives:
            nmb = self.nmb(perspective)
            mask = _tie_mask(nmb, atol=atol)
            tied = np.sum(mask, axis=1) > 1
            if policy is TiePolicy.ERROR and np.any(tied):
                raise PerspectiveError(
                    f"Tied optima detected for perspective {perspective!r}"
                )
            if policy is TiePolicy.FIRST:
                probabilities = np.mean(
                    np.eye(self.n_strategies)[np.argmax(mask, axis=1)], axis=0
                )
            else:
                probabilities = np.mean(
                    mask / np.sum(mask, axis=1, keepdims=True), axis=0
                )
            expected = np.mean(nmb, axis=0)
            ranks = _dense_descending_ranks(expected, atol=atol)
            for index, strategy in enumerate(self.strategies):
                rows.append(
                    PerspectiveAcceptabilityRow(
                        perspective=perspective,
                        strategy=strategy,
                        probability_optimal=float(probabilities[index]),
                        expected_net_benefit=float(expected[index]),
                        expected_value_rank=int(ranks[index]),
                        tie_policy=policy,
                    )
                )
        return rows

    def with_weighted_perspective(
        self,
        name: str,
        weights: Mapping[str, float],
        *,
        normalize: bool = True,
        allow_negative: bool = False,
    ) -> NetBenefitTensor:
        """Append a weighted combination of existing perspectives."""
        if name in self.perspectives:
            raise PerspectiveError(f"Perspective already exists: {name!r}")
        if not weights:
            raise PerspectiveError("weights must not be empty")
        vector = np.zeros(self.n_perspectives, dtype=np.float64)
        for perspective, raw_weight in weights.items():
            weight = float(raw_weight)
            if not np.isfinite(weight):
                raise PerspectiveError("weights must be finite")
            if weight < 0 and not allow_negative:
                raise PerspectiveError(
                    "weights must be non-negative unless allow_negative=True"
                )
            vector[self.perspective_index(perspective)] = weight
        total = float(np.sum(vector))
        if normalize:
            if total <= 0:
                raise PerspectiveError(
                    "normalized weighted perspective requires positive total weight"
                )
            vector = vector / total
        elif not allow_negative and total <= 0:
            raise PerspectiveError(
                "weighted perspective requires positive total weight"
            )
        mixture = np.tensordot(self.values, vector, axes=([2], [0]))[:, :, np.newaxis]
        return NetBenefitTensor(
            values=np.concatenate([self.values, mixture], axis=2),
            strategies=self.strategies,
            perspectives=(*tuple(self.perspectives), name),
            case_id=self.case_id,
            draw_ids=self.draw_ids,
            attrs={
                **self.attrs,
                "weighted_perspectives": {
                    **self.attrs.get("weighted_perspectives", {}),
                    name: {"weights": dict(weights), "normalize": normalize}
                },
            },
        )

    def mcda_feature_records(
        self,
        *,
        reference_perspective: str,
        target_perspectives: Iterable[str] | None = None,
        population: float | None = None,
    ) -> list[dict[str, Any]]:
        """Collect benefits, ranks, discordance and regret by lens and strategy."""
        targets = (
            tuple(target_perspectives)
            if target_perspectives is not None
            else tuple(self.perspectives)
        )
        acceptability = {
            (row.perspective, row.strategy): row
            for row in self.perspective_acceptability_frontier()
        }
        rows: list[dict[str, Any]] = []
        for perspective in targets:
            regret = self.evop(
                choose_under=reference_perspective,
                evaluate_under=perspective,
                population=population,
            )
            expected = self.expected_net_benefit(perspective)
            for index, strategy in enumerate(self.strategies):
                paf = acceptability[(perspective, strategy)]
                rows.append(
                    {
                        "case_id": self.case_id,
                        "reference_perspective": reference_perspective,
                        "evaluate_under": perspective,
                        "strategy": strategy,
                        "expected_net_benefit": float(expected[index]),
                        "probability_optimal": paf.probability_optimal,
                        "expected_value_rank": paf.expected_value_rank,
                        "evop_reference_to_evaluation_per_person": regret.per_person,
                        "evop_reference_to_evaluation_population": regret.population_value,
                        "perspective_discordance_probability": regret.discordance_probability,
                        "method_contract_version": METHOD_CONTRACT_VERSION,
                    }
                )
        return rows

    def perspective_mixture_frontier(
        self,
        *,
        left_perspective: str,
        right_perspective: str,
        grid_size: int = 101,
        tie_policy: TiePolicy | str = TiePolicy.SPLIT,
        atol: float = 1e-12,
    ) -> list[PerspectiveMixtureFrontierRow]:
        """Evaluate strategy performance on a regular two-perspective weight grid."""
        if grid_size < 2:
            raise PerspectiveError("grid_size must be at least 2")
        self.perspective_index(left_perspective)
        self.perspective_index(right_perspective)
        rows: list[PerspectiveMixtureFrontierRow] = []
        for right_weight in np.linspace(0.0, 1.0, grid_size):
            weighted = self.with_weighted_perspective(
                "__mixture__",
                {
                    left_perspective: 1.0 - float(right_weight),
                    right_perspective: float(right_weight),
                },
                normalize=False,
            )
            rows.extend(
                PerspectiveMixtureFrontierRow(
                    left_perspective=left_perspective,
                    right_perspective=right_perspective,
                    right_weight=float(right_weight),
                    strategy=row.strategy,
                    probability_optimal=row.probability_optimal,
                    expected_net_benefit=row.expected_net_benefit,
                    expected_value_rank=row.expected_value_rank,
                    tie_policy=row.tie_policy,
                )
                for row in weighted.perspective_acceptability_frontier(
                    tie_policy=tie_policy, atol=atol
                )
                if row.perspective == "__mixture__"
            )
        return rows

    def exact_perspective_frontier(
        self,
        *,
        left_perspective: str,
        right_perspective: str,
        atol: float = 1e-12,
    ) -> list[PerspectiveFrontierSegment]:
        """Return exact expected-value upper-envelope intervals over mixture weight."""
        left = self.expected_net_benefit(left_perspective)
        right = self.expected_net_benefit(right_perspective)
        slopes = right - left
        candidates = [0.0, 1.0]
        for i in range(self.n_strategies):
            for j in range(i + 1, self.n_strategies):
                denominator = float(slopes[i] - slopes[j])
                if np.isclose(denominator, 0.0, rtol=0.0, atol=atol):
                    continue
                weight = float((left[j] - left[i]) / denominator)
                if -atol <= weight <= 1.0 + atol:
                    candidates.append(min(1.0, max(0.0, weight)))
        points: list[float] = []
        for value in sorted(candidates):
            if not points or not np.isclose(value, points[-1], rtol=0.0, atol=atol):
                points.append(value)
        raw: list[PerspectiveFrontierSegment] = []
        for lower, upper in itertools.pairwise(points):
            midpoint = (lower + upper) / 2.0
            expected = left + slopes * midpoint
            maximum = float(np.max(expected))
            optimal = tuple(
                self.strategies[int(index)]
                for index in np.flatnonzero(
                    np.isclose(expected, maximum, rtol=0.0, atol=atol)
                )
            )
            raw.append(
                PerspectiveFrontierSegment(
                    left_perspective=left_perspective,
                    right_perspective=right_perspective,
                    lower_right_weight=float(lower),
                    upper_right_weight=float(upper),
                    optimal_strategies=optimal,
                    midpoint_expected_net_benefit=maximum,
                )
            )
        merged: list[PerspectiveFrontierSegment] = []
        for segment in raw:
            if (
                merged
                and merged[-1].optimal_strategies == segment.optimal_strategies
                and np.isclose(
                    merged[-1].upper_right_weight,
                    segment.lower_right_weight,
                    rtol=0.0,
                    atol=atol,
                )
            ):
                previous = merged[-1]
                midpoint = (
                    previous.lower_right_weight + segment.upper_right_weight
                ) / 2.0
                expected = left + slopes * midpoint
                merged[-1] = PerspectiveFrontierSegment(
                    left_perspective=left_perspective,
                    right_perspective=right_perspective,
                    lower_right_weight=previous.lower_right_weight,
                    upper_right_weight=segment.upper_right_weight,
                    optimal_strategies=segment.optimal_strategies,
                    midpoint_expected_net_benefit=float(np.max(expected)),
                )
            else:
                merged.append(segment)
        if not merged:
            expected = left
            maximum = float(np.max(expected))
            optimal = tuple(
                self.strategies[int(index)]
                for index in np.flatnonzero(
                    np.isclose(expected, maximum, rtol=0.0, atol=atol)
                )
            )
            merged.append(
                PerspectiveFrontierSegment(
                    left_perspective, right_perspective, 0.0, 1.0, optimal, maximum
                )
            )
        return merged

    def exact_switch_points(
        self, *, left_perspective: str, right_perspective: str, atol: float = 1e-12
    ) -> list[PerspectiveSwitchPoint]:
        """Extract changes between adjacent exact frontier segments."""
        segments = self.exact_perspective_frontier(
            left_perspective=left_perspective,
            right_perspective=right_perspective,
            atol=atol,
        )
        return [
            PerspectiveSwitchPoint(
                left_perspective=left_perspective,
                right_perspective=right_perspective,
                right_weight=previous.upper_right_weight,
                from_strategies=previous.optimal_strategies,
                to_strategies=current.optimal_strategies,
            )
            for previous, current in itertools.pairwise(segments)
            if previous.optimal_strategies != current.optimal_strategies
        ]


def expected_net_benefit(
    tensor: NetBenefitTensor, perspective: str
) -> NDArray[np.float64]:
    """Average each strategy's net benefit over draws."""
    return tensor.expected_net_benefit(perspective)


def evop(tensor: NetBenefitTensor, **kwargs: Any) -> PerspectiveRegret:
    """Evaluate directional regret with optional population scaling."""
    return tensor.evop(**kwargs)


def regret_matrix(tensor: NetBenefitTensor, **kwargs: Any) -> list[dict[str, Any]]:
    """Return directional regret records for ordered perspective pairs."""
    return tensor.regret_matrix(**kwargs)


def perspective_acceptability_frontier(
    tensor: NetBenefitTensor, **kwargs: Any
) -> list[PerspectiveAcceptabilityRow]:
    """Compute draw-level optimality probabilities and expected-value ranks."""
    return tensor.perspective_acceptability_frontier(**kwargs)


def with_weighted_perspective(
    tensor: NetBenefitTensor, name: str, weights: Mapping[str, float], **kwargs: Any
) -> NetBenefitTensor:
    """Append a weighted combination of existing perspectives."""
    return tensor.with_weighted_perspective(name, weights, **kwargs)


def perspective_mixture_frontier(
    tensor: NetBenefitTensor, **kwargs: Any
) -> list[PerspectiveMixtureFrontierRow]:
    """Evaluate strategy performance on a regular two-perspective weight grid."""
    return tensor.perspective_mixture_frontier(**kwargs)


def exact_perspective_frontier(
    tensor: NetBenefitTensor, **kwargs: Any
) -> list[PerspectiveFrontierSegment]:
    """Partition the mixture interval by optimal expected net benefit."""
    return tensor.exact_perspective_frontier(**kwargs)


def expected_value_switch_points(
    rows_or_tensor: Iterable[PerspectiveMixtureFrontierRow] | NetBenefitTensor,
    **kwargs: Any,
) -> list[PerspectiveSwitchPoint]:
    """Compatibility helper.

    Passing a tensor uses exact switch points. Passing grid rows retains the v5
    adjacent-grid approximation for migration only.
    """
    if isinstance(rows_or_tensor, NetBenefitTensor):
        return rows_or_tensor.exact_switch_points(**kwargs)
    materialised = sorted(
        rows_or_tensor,
        key=lambda row: (row.right_weight, row.expected_value_rank, row.strategy),
    )
    best = {
        row.right_weight: row for row in materialised if row.expected_value_rank == 1
    }
    ordered = [best[weight] for weight in sorted(best)]
    output: list[PerspectiveSwitchPoint] = []
    for previous, current in itertools.pairwise(ordered):
        if previous.strategy != current.strategy:
            output.append(
                PerspectiveSwitchPoint(
                    left_perspective=current.left_perspective,
                    right_perspective=current.right_perspective,
                    right_weight=(previous.right_weight + current.right_weight) / 2.0,
                    from_strategies=(previous.strategy,),
                    to_strategies=(current.strategy,),
                )
            )
    return output


def mcda_feature_records(
    tensor: NetBenefitTensor, **kwargs: Any
) -> list[dict[str, Any]]:
    """Collect benefits, ranks, discordance and regret by lens and strategy."""
    return tensor.mcda_feature_records(**kwargs)


def tensor_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    value_column: str = "net_benefit",
    draw_column: str = "draw",
    strategy_column: str = "strategy",
    perspective_column: str = "perspective",
    case_id: str | None = None,
) -> NetBenefitTensor:
    """Build a dense labelled tensor from long-form net benefit records."""
    rows = list(records)
    if not rows:
        raise PerspectiveError("Cannot build tensor from empty records")
    try:
        draws = tuple(dict.fromkeys(str(row[draw_column]) for row in rows))
        strategies = tuple(dict.fromkeys(str(row[strategy_column]) for row in rows))
        perspectives = tuple(
            dict.fromkeys(str(row[perspective_column]) for row in rows)
        )
    except KeyError as exc:
        raise PerspectiveError(
            f"Missing required record column: {exc.args[0]}"
        ) from exc
    draw_index = {name: index for index, name in enumerate(draws)}
    strategy_index = {name: index for index, name in enumerate(strategies)}
    perspective_index = {name: index for index, name in enumerate(perspectives)}
    values = np.full(
        (len(draws), len(strategies), len(perspectives)), np.nan, dtype=np.float64
    )
    for row in rows:
        i = draw_index[str(row[draw_column])]
        j = strategy_index[str(row[strategy_column])]
        k = perspective_index[str(row[perspective_column])]
        if np.isfinite(values[i, j, k]):
            raise PerspectiveError("Duplicate draw/strategy/perspective record")
        values[i, j, k] = float(row[value_column])
    if np.isnan(values).any():
        raise PerspectiveError(
            f"Records are not dense; {int(np.isnan(values).sum())} values missing"
        )
    return NetBenefitTensor(
        values=values,
        strategies=strategies,
        perspectives=perspectives,
        case_id=case_id,
        draw_ids=draws,
    )
