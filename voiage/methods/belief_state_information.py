"""Exact finite belief-state and intervention-aware information value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_CHRONOLOGY = ["control", "transition", "observe", "update"]
_MAX_EXACT_BELLMAN_EXPANSIONS = 50_000
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "analysis_id",
    "analysis_type",
    "method_maturity",
    "value_unit",
    "time_unit",
    "objective_direction",
    "horizon",
    "discount_factor",
    "chronology",
    "policy_class",
    "stopping",
    "latent_states",
    "control_actions",
    "observations",
    "transition_model",
    "rewards",
    "sensors",
    "constraints",
    "tolerances",
}


@dataclass(frozen=True)
class BeliefStateInformationResult:
    """Portable result for exact finite belief-state information value."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


@dataclass(frozen=True)
class _Evaluation:
    net: float
    gross: float
    sensing_cost: float
    tree: dict[str, Any]


@dataclass(frozen=True)
class _Model:
    payload: dict[str, Any]
    states: tuple[str, ...]
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    sensors: tuple[str, ...]
    null_sensor: str
    initial_belief: dict[str, float]
    transition: dict[str, dict[str, dict[str, float]]]
    rewards: dict[str, dict[str, float]]
    likelihood: dict[str, dict[str, dict[str, dict[str, float]]]]
    sensor_costs: dict[str, float]
    allowed_actions: dict[int, tuple[str, ...]]
    allowed_sensors: dict[str, tuple[str, ...]]
    horizon: int
    discount: float
    sign: float
    absolute_tie: float
    relative_tie: float
    probability_tolerance: float
    estimated_bellman_expansions: int


def belief_state_information_value(
    specification: Mapping[str, object],
) -> BeliefStateInformationResult:
    """Evaluate an exact finite control-transition-observe-update problem.

    The adaptive Bellman policy is compared with a matched no-information
    policy. The result reports action-dependent-learning and dual-control
    diagnostics, but deliberately does not manufacture a unique additive
    dual-control value.
    """
    try:
        payload = cast(
            "dict[str, Any]",
            json.loads(json.dumps(specification, ensure_ascii=False)),
        )
        model = _validate_and_build(payload)
        result = _evaluate(model)
        validate_belief_state_information_result(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return BeliefStateInformationResult(result)


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast("dict[str, Any]", value)


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty array")
    return cast("list[Any]", value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _probabilities(
    raw: object,
    ids: Sequence[str],
    name: str,
    absolute: float,
) -> dict[str, float]:
    mapping = _require_mapping(raw, name)
    if set(mapping) != set(ids):
        raise ValueError(f"{name} must contain all latent states or observations")
    result = {
        item_id: _finite(mapping[item_id], f"{name}.{item_id}") for item_id in ids
    }
    if any(value < 0.0 or value > 1.0 for value in result.values()):
        raise ValueError(f"{name} probabilities must be between zero and one")
    if not math.isclose(math.fsum(result.values()), 1.0, abs_tol=absolute, rel_tol=0.0):
        raise ValueError(f"{name} probabilities must sum to one")
    return result


def _unique_ids(records: object, field: str, name: str) -> tuple[str, ...]:
    items = _require_list(records, name)
    identifiers: list[str] = []
    for item in items:
        record = _require_mapping(item, f"{name} entry")
        expected_fields = (
            {field, "initial_probability"}
            if name == "latent_states"
            else {field}
            if name in {"control_actions", "observations"}
            else None
        )
        if expected_fields is not None and set(record) != expected_fields:
            raise ValueError(f"{name} entry fields must be strict")
        identifier = record.get(field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{name}.{field} must be a non-empty string")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} identifiers must be unique")
    return tuple(sorted(identifiers))


def _validate_and_build(payload: dict[str, Any]) -> _Model:
    unknown = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    required = _TOP_LEVEL_FIELDS
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    if payload["schema_version"] != "1.0.0":
        raise ValueError("schema_version must be 1.0.0")
    if payload["analysis_type"] != "belief_state_information":
        raise ValueError("analysis_type must be belief_state_information")
    if payload["method_maturity"] != "experimental":
        raise ValueError("method_maturity must be experimental")
    for field in ("analysis_id", "value_unit", "time_unit"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if payload["objective_direction"] not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be maximize or minimize")
    if payload["chronology"] != _CHRONOLOGY:
        raise ValueError("chronology must be control-transition-observe-update")
    if payload["policy_class"] != "deterministic_markov_belief_policy":
        raise ValueError("policy_class must be deterministic_markov_belief_policy")
    stopping = _require_mapping(payload["stopping"], "stopping")
    if stopping != {"kind": "fixed_horizon"}:
        raise ValueError("stopping must declare fixed_horizon")
    horizon_value = payload["horizon"]
    if isinstance(horizon_value, bool) or not isinstance(horizon_value, int):
        raise TypeError("horizon must be an integer")
    horizon = int(horizon_value)
    if horizon < 1 or horizon > 12:
        raise ValueError("horizon must be between 1 and 12")
    discount = _finite(payload["discount_factor"], "discount_factor")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount_factor must be in (0, 1]")
    tolerances = _require_mapping(payload["tolerances"], "tolerances")
    if set(tolerances) != {"absolute_tie", "relative_tie", "probability"}:
        raise ValueError(
            "tolerances must declare absolute_tie, relative_tie, probability"
        )
    absolute = _finite(tolerances["absolute_tie"], "absolute_tie")
    relative = _finite(tolerances["relative_tie"], "relative_tie")
    probability_tolerance = _finite(tolerances["probability"], "probability")
    if min(absolute, relative, probability_tolerance) < 0.0:
        raise ValueError("tolerances must be nonnegative")
    if max(absolute, relative, probability_tolerance) > 1e-6:
        raise ValueError("tolerances must not exceed 1e-6")

    states = _unique_ids(payload["latent_states"], "state_id", "latent_states")
    actions = _unique_ids(payload["control_actions"], "action_id", "control_actions")
    observations = _unique_ids(
        payload["observations"], "observation_id", "observations"
    )
    initial = {
        cast("str", item["state_id"]): _finite(
            item.get("initial_probability"), "initial_probability"
        )
        for item in _require_list(payload["latent_states"], "latent_states")
    }
    if not math.isclose(
        math.fsum(initial.values()),
        1.0,
        abs_tol=probability_tolerance,
        rel_tol=0.0,
    ):
        raise ValueError("initial probabilities must sum to one")
    if any(value < 0.0 or value > 1.0 for value in initial.values()):
        raise ValueError("initial probabilities must be between zero and one")

    transition_raw = _require_mapping(payload["transition_model"], "transition_model")
    if set(transition_raw) != set(actions):
        raise ValueError("transition_model must contain all control actions")
    transition: dict[str, dict[str, dict[str, float]]] = {}
    for action in actions:
        by_state = _require_mapping(
            transition_raw[action], f"transition_model.{action}"
        )
        if set(by_state) != set(states):
            raise ValueError("transition_model must contain all latent states")
        transition[action] = {
            state: _probabilities(
                by_state[state],
                states,
                f"transition_model.{action}.{state}",
                probability_tolerance,
            )
            for state in states
        }

    rewards_raw = _require_mapping(payload["rewards"], "rewards")
    if set(rewards_raw) != set(actions):
        raise ValueError("rewards must contain all control actions")
    rewards: dict[str, dict[str, float]] = {}
    for action in actions:
        by_state = _require_mapping(rewards_raw[action], f"rewards.{action}")
        if set(by_state) != set(states):
            raise ValueError("rewards must contain all latent states")
        rewards[action] = {
            state: _finite(by_state[state], f"rewards.{action}.{state}")
            for state in states
        }

    sensor_records = _require_list(payload["sensors"], "sensors")
    sensors = _unique_ids(sensor_records, "sensor_id", "sensors")
    nulls = [
        cast("str", item["sensor_id"])
        for item in sensor_records
        if _require_mapping(item, "sensor").get("null_sensor") is True
    ]
    if len(nulls) != 1:
        raise ValueError("exactly one null sensor must be declared")
    null_sensor = nulls[0]
    likelihood: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    sensor_costs: dict[str, float] = {}
    for item in sensor_records:
        record = _require_mapping(item, "sensor")
        if set(record) != {"sensor_id", "null_sensor", "cost", "likelihood_by_control"}:
            raise ValueError("sensor fields must be strict")
        sensor_id = cast("str", record["sensor_id"])
        if type(record["null_sensor"]) is not bool:
            raise TypeError("sensor null_sensor must be boolean")
        cost = _finite(record["cost"], f"sensor {sensor_id} cost")
        if cost < 0.0:
            raise ValueError("sensor costs must be nonnegative")
        if sensor_id == null_sensor and cost != 0.0:
            raise ValueError("null sensor cost must be zero")
        sensor_costs[sensor_id] = cost
        by_control = _require_mapping(
            record["likelihood_by_control"], f"sensor {sensor_id} likelihood"
        )
        if set(by_control) != set(actions):
            raise ValueError("sensor likelihood must contain all control actions")
        likelihood[sensor_id] = {}
        for action in actions:
            by_state = _require_mapping(
                by_control[action], f"sensor {sensor_id}.{action}"
            )
            if set(by_state) != set(states):
                raise ValueError("sensor likelihood must contain all latent states")
            likelihood[sensor_id][action] = {
                state: _probabilities(
                    by_state[state],
                    observations,
                    f"sensor {sensor_id}.{action}.{state}",
                    probability_tolerance,
                )
                for state in states
            }
    for action in actions:
        reference = likelihood[null_sensor][action][states[0]]
        if any(
            any(
                not math.isclose(
                    likelihood[null_sensor][action][state][observation],
                    reference[observation],
                    abs_tol=probability_tolerance,
                    rel_tol=0.0,
                )
                for observation in observations
            )
            for state in states[1:]
        ):
            raise ValueError("null sensor must be state independent")

    constraints = _require_mapping(payload["constraints"], "constraints")
    if set(constraints) != {
        "allowed_control_action_ids_by_stage",
        "allowed_sensor_ids_by_control",
    }:
        raise ValueError("constraints fields must be strict")
    stages = _require_mapping(
        constraints["allowed_control_action_ids_by_stage"],
        "allowed_control_action_ids_by_stage",
    )
    if set(stages) != {str(stage) for stage in range(horizon)}:
        raise ValueError("allowed controls must cover every stage")
    allowed_actions: dict[int, tuple[str, ...]] = {}
    for stage in range(horizon):
        allowed = tuple(sorted(cast("list[str]", stages[str(stage)])))
        if (
            not allowed
            or len(set(allowed)) != len(allowed)
            or not set(allowed) <= set(actions)
        ):
            raise ValueError("allowed controls must be non-empty known unique actions")
        allowed_actions[stage] = allowed
    sensors_by_control = _require_mapping(
        constraints["allowed_sensor_ids_by_control"],
        "allowed_sensor_ids_by_control",
    )
    if set(sensors_by_control) != set(actions):
        raise ValueError("allowed sensors must cover all control actions")
    allowed_sensors: dict[str, tuple[str, ...]] = {}
    for action in actions:
        allowed = tuple(sorted(cast("list[str]", sensors_by_control[action])))
        if (
            not allowed
            or null_sensor not in allowed
            or len(set(allowed)) != len(allowed)
            or not set(allowed) <= set(sensors)
        ):
            raise ValueError("allowed sensors must be known, unique and include null")
        allowed_sensors[action] = allowed

    estimated_expansions = _estimate_bellman_expansions(
        horizon=horizon,
        states=len(states),
        observations=len(observations),
        allowed_actions=allowed_actions,
        allowed_sensors=allowed_sensors,
    )
    if estimated_expansions > _MAX_EXACT_BELLMAN_EXPANSIONS:
        message = f"exact Bellman expansion estimate {estimated_expansions} exceeds the supported budget {_MAX_EXACT_BELLMAN_EXPANSIONS}"
        message += "; reduce the horizon or finite spaces"
        raise ValueError(message)

    return _Model(
        payload=payload,
        states=states,
        actions=actions,
        observations=observations,
        sensors=sensors,
        null_sensor=null_sensor,
        initial_belief={state: initial[state] for state in states},
        transition=transition,
        rewards=rewards,
        likelihood=likelihood,
        sensor_costs=sensor_costs,
        allowed_actions=allowed_actions,
        allowed_sensors=allowed_sensors,
        horizon=horizon,
        discount=discount,
        sign=1.0 if payload["objective_direction"] == "maximize" else -1.0,
        absolute_tie=absolute,
        relative_tie=relative,
        probability_tolerance=probability_tolerance,
        estimated_bellman_expansions=estimated_expansions,
    )


def _estimate_bellman_expansions(
    *,
    horizon: int,
    states: int,
    observations: int,
    allowed_actions: Mapping[int, Sequence[str]],
    allowed_sensors: Mapping[str, Sequence[str]],
) -> int:
    """Return a conservative bound for every recursive evaluator invocation.

    The bound deliberately ignores adaptive-cache hits. It therefore remains
    safe when numerically equal beliefs have different floating-point
    representations, while memoization can still reduce actual work. The
    fully observed comparator branches over every declared latent-state entry,
    including zero-probability transitions, so its state factor must be counted
    independently of the observation-space factor used by the belief solver.
    """

    def recursive_calls(branch_factors: Sequence[int]) -> int:
        calls = 1
        for branch_factor in reversed(branch_factors[:-1]):
            calls = 1 + branch_factor * calls
        return calls

    adaptive_factors = [
        observations
        * sum(len(allowed_sensors[action]) for action in allowed_actions[stage])
        for stage in range(horizon)
    ]
    no_information_factors = [len(allowed_actions[stage]) for stage in range(horizon)]
    fully_observed_factors = [
        states * len(allowed_actions[stage]) for stage in range(horizon)
    ]

    adaptive_full = recursive_calls(adaptive_factors)
    adaptive_horizon_curve = sum(
        recursive_calls(adaptive_factors[:partial_horizon])
        for partial_horizon in range(1, horizon + 1)
    )
    adaptive_myopic = 1
    conditional = (
        adaptive_factors[0] * recursive_calls(adaptive_factors[1:])
        if horizon > 1
        else 0
    )

    no_information_full = recursive_calls(no_information_factors)
    no_information_horizon_curve = sum(
        recursive_calls(no_information_factors[:partial_horizon])
        for partial_horizon in range(1, horizon + 1)
    )
    no_information_myopic = 1

    fully_observed = states * recursive_calls(fully_observed_factors)
    return (
        adaptive_full
        + adaptive_horizon_curve
        + adaptive_myopic
        + conditional
        + no_information_full
        + no_information_horizon_curve
        + no_information_myopic
        + fully_observed
    )


def _expected_reward(model: _Model, belief: Mapping[str, float], action: str) -> float:
    return math.fsum(
        belief[state] * model.rewards[action][state] for state in model.states
    )


def _predict(
    model: _Model, belief: Mapping[str, float], action: str
) -> dict[str, float]:
    return {
        next_state: math.fsum(
            belief[state] * model.transition[action][state][next_state]
            for state in model.states
        )
        for next_state in model.states
    }


def _observe(
    model: _Model,
    predictive: Mapping[str, float],
    action: str,
    sensor: str,
) -> list[tuple[str, float, dict[str, float]]]:
    branches: list[tuple[str, float, dict[str, float]]] = []
    for observation in model.observations:
        probability = math.fsum(
            predictive[state] * model.likelihood[sensor][action][state][observation]
            for state in model.states
        )
        if probability <= 0.0:
            continue
        posterior = {
            state: predictive[state]
            * model.likelihood[sensor][action][state][observation]
            / probability
            for state in model.states
        }
        branches.append((observation, probability, posterior))
    if not math.isclose(
        math.fsum(item[1] for item in branches),
        1.0,
        abs_tol=model.probability_tolerance,
        rel_tol=0.0,
    ):
        raise ArithmeticError("observation branches must sum to one")
    return branches


def _ties(model: _Model, values: Mapping[str, float]) -> list[str]:
    maximum = max(values.values())
    return sorted(
        key
        for key, value in values.items()
        if math.isclose(
            value,
            maximum,
            abs_tol=model.absolute_tie,
            rel_tol=model.relative_tie,
        )
    )


def _adaptive(
    model: _Model,
    stage: int,
    belief: dict[str, float],
    cache: dict[tuple[int, tuple[float, ...]], _Evaluation] | None = None,
) -> _Evaluation:
    if cache is None:
        cache = {}
    cache_key = (stage, tuple(belief[state] for state in model.states))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    action_evaluations: dict[str, _Evaluation] = {}
    for action in model.allowed_actions[stage]:
        immediate_original = _expected_reward(model, belief, action)
        immediate_oriented = model.sign * immediate_original
        predictive = _predict(model, belief, action)
        sensor_evaluations: dict[str, _Evaluation] = {}
        for sensor in model.allowed_sensors[action]:
            branches = _observe(model, predictive, action, sensor)
            branch_records: list[dict[str, Any]] = []
            future_net = 0.0
            future_gross = 0.0
            future_cost = 0.0
            for observation, probability, posterior in branches:
                child = (
                    _adaptive(model, stage + 1, posterior, cache)
                    if stage + 1 < model.horizon
                    else _Evaluation(0.0, 0.0, 0.0, {})
                )
                future_net += probability * child.net
                future_gross += probability * child.gross
                future_cost += probability * child.sensing_cost
                branch_records.append(
                    {
                        "observation_id": observation,
                        "probability": probability,
                        "posterior_belief": posterior,
                        "continuation": child.tree or None,
                    }
                )
            cost = model.sensor_costs[sensor]
            sensor_evaluations[sensor] = _Evaluation(
                net=immediate_oriented + model.discount * future_net - cost,
                gross=immediate_original + model.discount * future_gross,
                sensing_cost=cost + model.discount * future_cost,
                tree={
                    "sensor_id": sensor,
                    "branches": branch_records,
                    "predictive_belief": predictive,
                },
            )
        sensor_values = {key: value.net for key, value in sensor_evaluations.items()}
        sensor_ties = _ties(model, sensor_values)
        selected_sensor = sensor_ties[0]
        selected = sensor_evaluations[selected_sensor]
        action_evaluations[action] = _Evaluation(
            selected.net,
            selected.gross,
            selected.sensing_cost,
            {
                "stage": stage,
                "belief": belief,
                "control_action_id": action,
                "sensor_choice_tie": sensor_ties,
                "selected_sensor": selected_sensor,
                "predictive_belief": selected.tree["predictive_belief"],
                "branches": selected.tree["branches"],
            },
        )
    action_values = {key: value.net for key, value in action_evaluations.items()}
    action_ties = _ties(model, action_values)
    selected_action = action_ties[0]
    selected = action_evaluations[selected_action]
    tree = dict(selected.tree)
    tree.update(
        {
            "control_choice_tie": action_ties,
            "selected_control": selected_action,
            "chronology": list(_CHRONOLOGY),
        }
    )
    tree.pop("control_action_id")
    evaluation = _Evaluation(selected.net, selected.gross, selected.sensing_cost, tree)
    cache[cache_key] = evaluation
    return evaluation


def _no_information(model: _Model, stage: int, belief: dict[str, float]) -> float:
    values: dict[str, float] = {}
    for action in model.allowed_actions[stage]:
        reward = model.sign * _expected_reward(model, belief, action)
        predictive = _predict(model, belief, action)
        continuation = (
            _no_information(model, stage + 1, predictive)
            if stage + 1 < model.horizon
            else 0.0
        )
        values[action] = reward + model.discount * continuation
    return max(values.values())


def _fully_observed(model: _Model, stage: int, state: str) -> float:
    values: list[float] = []
    for action in model.allowed_actions[stage]:
        continuation = 0.0
        if stage + 1 < model.horizon:
            continuation = math.fsum(
                probability * _fully_observed(model, stage + 1, next_state)
                for next_state, probability in model.transition[action][state].items()
            )
        values.append(
            model.sign * model.rewards[action][state] + model.discount * continuation
        )
    return max(values)


def _conditional_sensing(
    model: _Model,
    cache: dict[tuple[int, tuple[float, ...]], _Evaluation],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    belief = model.initial_belief
    for action in model.allowed_actions[0]:
        predictive = _predict(model, belief, action)
        net_values: dict[str, float] = {}
        gross_values: dict[str, float] = {}
        for sensor in model.allowed_sensors[action]:
            future_net = 0.0
            future_gross_oriented = 0.0
            for _, probability, posterior in _observe(
                model, predictive, action, sensor
            ):
                if model.horizon > 1:
                    child = _adaptive(model, 1, posterior, cache)
                    future_net += probability * child.net
                    future_gross_oriented += probability * model.sign * child.gross
            gross = model.discount * future_gross_oriented
            net = model.discount * future_net - model.sensor_costs[sensor]
            gross_values[sensor] = gross
            net_values[sensor] = net
        null_net = net_values[model.null_sensor]
        sensor_records = [
            {
                "sensor_id": sensor,
                "gross_value": gross_values[sensor],
                "sensing_cost": model.sensor_costs[sensor],
                "net_value": net_values[sensor],
                "net_increment_vs_null": net_values[sensor] - null_net,
            }
            for sensor in sorted(net_values)
        ]
        records.append({"control_action_id": action, "sensors": sensor_records})
    return records


def _martingale_residual(model: _Model, tree: Mapping[str, Any]) -> float:
    predictive = cast("Mapping[str, float]", tree["predictive_belief"])
    branches = cast("list[Mapping[str, Any]]", tree["branches"])
    residual = max(
        abs(
            math.fsum(
                float(branch["probability"])
                * float(cast("Mapping[str, float]", branch["posterior_belief"])[state])
                for branch in branches
            )
            - predictive[state]
        )
        for state in model.states
    )
    for branch in branches:
        child = branch["continuation"]
        if isinstance(child, dict):
            residual = max(residual, _martingale_residual(model, child))
    return residual


def _state_informative(model: _Model, sensor: str, action: str) -> bool:
    reference = model.likelihood[sensor][action][model.states[0]]
    return any(
        any(
            not math.isclose(
                model.likelihood[sensor][action][state][observation],
                reference[observation],
                abs_tol=model.probability_tolerance,
                rel_tol=0.0,
            )
            for observation in model.observations
        )
        for state in model.states[1:]
    )


def _selected_policy_uses_learning(tree: Mapping[str, Any]) -> bool:
    continuations = [
        branch["continuation"]
        for branch in cast("list[Mapping[str, Any]]", tree["branches"])
        if isinstance(branch["continuation"], dict)
    ]
    if (
        len(
            {
                cast("str", continuation["selected_control"])
                for continuation in continuations
            }
        )
        > 1
    ):
        return True
    return any(
        _selected_policy_uses_learning(cast("Mapping[str, Any]", continuation))
        for continuation in continuations
    )


def _action_dependent_learning(model: _Model, usable_response: bool) -> bool:
    if not usable_response or model.horizon < 2:
        return False
    for stage in range(model.horizon - 1):
        for sensor in model.sensors:
            usable_actions = [
                action
                for action in model.allowed_actions[stage]
                if sensor in model.allowed_sensors[action]
            ]
            for index, left in enumerate(usable_actions):
                for right in usable_actions[index + 1 :]:
                    if (
                        _state_informative(model, sensor, left)
                        or _state_informative(model, sensor, right)
                    ) and any(
                        any(
                            not math.isclose(
                                model.likelihood[sensor][left][state][observation],
                                model.likelihood[sensor][right][state][observation],
                                abs_tol=model.probability_tolerance,
                                rel_tol=0.0,
                            )
                            for observation in model.observations
                        )
                        for state in model.states
                    ):
                        return True
    return False


def _evaluate(model: _Model) -> dict[str, Any]:
    adaptive_cache: dict[tuple[int, tuple[float, ...]], _Evaluation] = {}
    adaptive = _adaptive(model, 0, model.initial_belief, adaptive_cache)
    no_information_oriented = _no_information(model, 0, model.initial_belief)
    fully_observed_oriented = math.fsum(
        model.initial_belief[state] * _fully_observed(model, 0, state)
        for state in model.states
    )
    gross_oriented = model.sign * adaptive.gross
    gross_voi = gross_oriented - no_information_oriented
    net_voi = adaptive.net - no_information_oriented
    if net_voi < -model.absolute_tie:
        raise ArithmeticError(
            "null-sensor reduction failed: information value is negative"
        )
    one_stage = replace(
        model,
        horizon=1,
        allowed_actions={0: model.allowed_actions[0]},
    )
    myopic = _adaptive(one_stage, 0, one_stage.initial_belief).net - _no_information(
        one_stage, 0, one_stage.initial_belief
    )
    horizon_values: list[dict[str, Any]] = []
    for horizon in range(1, model.horizon + 1):
        partial = replace(
            model,
            horizon=horizon,
            allowed_actions={
                stage: model.allowed_actions[stage] for stage in range(horizon)
            },
        )
        closed = _adaptive(partial, 0, partial.initial_belief).net
        baseline = _no_information(partial, 0, partial.initial_belief)
        horizon_values.append(
            {
                "horizon": horizon,
                "closed_loop_net": closed / model.sign,
                "no_information": baseline / model.sign,
                "net_information_value": closed - baseline,
            }
        )
    martingale_residual = _martingale_residual(model, adaptive.tree)
    usable_learning_response = _selected_policy_uses_learning(adaptive.tree)
    action_learning = _action_dependent_learning(model, usable_learning_response)
    transition_dependent = (
        len(
            {
                json.dumps(model.transition[action], sort_keys=True)
                for action in model.actions
            }
        )
        > 1
    )
    conditional = _conditional_sensing(model, adaptive_cache)
    null_reduction = all(
        math.isclose(
            next(
                sensor["net_increment_vs_null"]
                for sensor in record["sensors"]
                if sensor["sensor_id"] == model.null_sensor
            ),
            0.0,
            abs_tol=model.absolute_tie,
        )
        for record in conditional
    )
    return {
        "schema_version": "1.0.0",
        "analysis_id": model.payload["analysis_id"],
        "analysis_type": "belief_state_information_result",
        "method_maturity": "experimental",
        "value_unit": model.payload["value_unit"],
        "time_unit": model.payload["time_unit"],
        "objective_direction": model.payload["objective_direction"],
        "chronology": list(_CHRONOLOGY),
        "horizon": model.horizon,
        "discount_factor": model.discount,
        "policy_class": model.payload["policy_class"],
        "values": {
            "closed_loop_gross": adaptive.gross,
            "expected_sensing_cost": adaptive.sensing_cost,
            "closed_loop_net": adaptive.net / model.sign,
            "no_information": no_information_oriented / model.sign,
            "gross_information_value": gross_voi,
            "net_information_value": max(0.0, net_voi),
            "myopic_information_value": max(0.0, myopic),
            "nonmyopic_information_value": max(0.0, net_voi),
            "nonmyopic_minus_myopic": net_voi - myopic,
            "fully_observed_value": fully_observed_oriented / model.sign,
            "partial_observability_regret": fully_observed_oriented - gross_oriented,
        },
        "value_by_horizon": horizon_values,
        "conditional_sensing_values": conditional,
        "policy_tree": adaptive.tree,
        "stopping": {
            "kind": "fixed_horizon",
            "reason": "horizon_reached",
            "stage": model.horizon,
        },
        "approximation_bounds": {
            "lower": max(0.0, net_voi),
            "upper": max(0.0, net_voi),
            "gap": 0.0,
        },
        "assurance": {
            "solver": "exact_finite_horizon_bellman_enumeration",
            "estimator": "deterministic_exact",
            "exact_enumeration": True,
            "approximation_used": False,
            "posterior_martingale_verified": (
                martingale_residual <= model.probability_tolerance
            ),
            "posterior_martingale_max_residual": martingale_residual,
            "null_sensor_reduction_verified": null_reduction,
            "no_information_reduction_verified": net_voi >= -model.absolute_tie,
            "complete_ties_reported": True,
            "action_dependent_transition": transition_dependent,
            "action_dependent_learning": action_learning,
            "usable_downstream_learning_response": usable_learning_response,
            "dual_control_diagnostic": action_learning,
            "unique_additive_dual_control_value_claimed": False,
            "estimated_bellman_expansions": model.estimated_bellman_expansions,
            "exact_enumeration_budget": _MAX_EXACT_BELLMAN_EXPANSIONS,
        },
        "language_dispositions": {
            "python": "executable-experimental",
            "rust": "unsupported",
            "r": "unsupported",
            "julia": "unsupported",
            "mojo": "external-boundary",
        },
        "limitations": [
            "finite latent, action and observation spaces only",
            "fixed-horizon deterministic Markov belief policies only",
            "no unique additive dual-control value is claimed",
            "scientific review and cross-language parity remain pending",
        ],
    }


def _strict_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    mapping = _require_mapping(value, name)
    if set(mapping) != expected:
        raise ValueError(f"{name} fields must be strict")
    return mapping


def _validate_probability_map(value: object, name: str) -> dict[str, Any]:
    mapping = _require_mapping(value, name)
    if not mapping:
        raise ValueError(f"{name} must be non-empty")
    probabilities = [_finite(item, name) for item in mapping.values()]
    if any(item < 0.0 or item > 1.0 for item in probabilities) or not math.isclose(
        math.fsum(probabilities), 1.0, abs_tol=1e-6, rel_tol=0.0
    ):
        raise ValueError(f"{name} must be a probability distribution")
    return mapping


def _validate_policy_tree(value: object, state_ids: set[str]) -> None:
    tree = _strict_keys(
        value,
        {
            "stage",
            "belief",
            "control_choice_tie",
            "selected_control",
            "sensor_choice_tie",
            "selected_sensor",
            "predictive_belief",
            "branches",
            "chronology",
        },
        "policy_tree",
    )
    if isinstance(tree["stage"], bool) or not isinstance(tree["stage"], int):
        raise TypeError("policy_tree.stage must be an integer")
    if tree["chronology"] != _CHRONOLOGY:
        raise ValueError("policy_tree chronology must match the contract")
    belief = _validate_probability_map(tree["belief"], "policy_tree.belief")
    predictive = _validate_probability_map(
        tree["predictive_belief"], "policy_tree.predictive_belief"
    )
    if set(belief) != state_ids or set(predictive) != state_ids:
        raise ValueError("policy-tree belief state IDs must remain constant")
    control_tie = _require_list(tree["control_choice_tie"], "control_choice_tie")
    sensor_tie = _require_list(tree["sensor_choice_tie"], "sensor_choice_tie")
    if (
        any(not isinstance(item, str) or not item for item in control_tie + sensor_tie)
        or len(set(control_tie)) != len(control_tie)
        or len(set(sensor_tie)) != len(sensor_tie)
    ):
        raise ValueError("policy-tree ties must contain unique non-empty strings")
    if tree["selected_control"] not in control_tie:
        raise ValueError("selected control must belong to the reported tie")
    if tree["selected_sensor"] not in sensor_tie:
        raise ValueError("selected sensor must belong to the reported tie")
    branches = _require_list(tree["branches"], "policy_tree.branches")
    branch_probability = 0.0
    observation_ids: set[str] = set()
    for raw_branch in branches:
        branch = _strict_keys(
            raw_branch,
            {"observation_id", "probability", "posterior_belief", "continuation"},
            "policy_tree branch",
        )
        observation_id = branch["observation_id"]
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("branch observation IDs must be non-empty strings")
        if observation_id in observation_ids:
            raise ValueError("policy-tree branch observation IDs must be unique")
        observation_ids.add(observation_id)
        probability = _finite(branch["probability"], "branch probability")
        if probability <= 0.0 or probability > 1.0:
            raise ValueError("reported branches must have positive probability")
        branch_probability += probability
        posterior = _validate_probability_map(
            branch["posterior_belief"], "branch posterior"
        )
        if set(posterior) != state_ids:
            raise ValueError("posterior state IDs must match the root belief")
        continuation = branch["continuation"]
        if continuation is not None:
            _validate_policy_tree(continuation, state_ids)
            if cast("Mapping[str, Any]", continuation)["stage"] != tree["stage"] + 1:
                raise ValueError("policy-tree continuation stages must be consecutive")
    if not math.isclose(branch_probability, 1.0, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError("policy-tree branch probabilities must sum to one")


def _policy_martingale_residual(tree: Mapping[str, Any], state_ids: set[str]) -> float:
    predictive = cast("Mapping[str, Any]", tree["predictive_belief"])
    branches = cast("list[Mapping[str, Any]]", tree["branches"])
    residual = max(
        abs(
            math.fsum(
                float(branch["probability"])
                * float(cast("Mapping[str, Any]", branch["posterior_belief"])[state])
                for branch in branches
            )
            - float(predictive[state])
        )
        for state in state_ids
    )
    for branch in branches:
        continuation = branch["continuation"]
        if isinstance(continuation, dict):
            residual = max(
                residual,
                _policy_martingale_residual(continuation, state_ids),
            )
    return residual


def validate_belief_state_information_result(payload: Mapping[str, object]) -> None:
    """Fail closed on structural or numerical drift in a result envelope."""
    result = _strict_keys(
        payload,
        {
            "schema_version",
            "analysis_id",
            "analysis_type",
            "method_maturity",
            "value_unit",
            "time_unit",
            "objective_direction",
            "chronology",
            "horizon",
            "discount_factor",
            "policy_class",
            "values",
            "value_by_horizon",
            "conditional_sensing_values",
            "policy_tree",
            "stopping",
            "approximation_bounds",
            "assurance",
            "language_dispositions",
            "limitations",
        },
        "belief-state information result",
    )
    if (
        result["schema_version"] != "1.0.0"
        or result["analysis_type"] != "belief_state_information_result"
        or result["method_maturity"] != "experimental"
        or result["chronology"] != _CHRONOLOGY
        or result["policy_class"] != "deterministic_markov_belief_policy"
    ):
        raise ValueError("result envelope constants do not match the v1 contract")
    for field in ("analysis_id", "value_unit", "time_unit"):
        if not isinstance(result[field], str) or not result[field]:
            raise ValueError(f"result {field} must be a non-empty string")
    direction = result["objective_direction"]
    if direction not in {"maximize", "minimize"}:
        raise ValueError("result objective direction must be maximize or minimize")
    horizon = result["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise TypeError("result horizon must be a positive integer")
    discount = _finite(result["discount_factor"], "result discount factor")
    if not 0.0 < discount <= 1.0:
        raise ValueError("result discount factor must be in (0, 1]")
    values = _strict_keys(
        result["values"],
        {
            "closed_loop_gross",
            "expected_sensing_cost",
            "closed_loop_net",
            "no_information",
            "gross_information_value",
            "net_information_value",
            "myopic_information_value",
            "nonmyopic_information_value",
            "nonmyopic_minus_myopic",
            "fully_observed_value",
            "partial_observability_regret",
        },
        "result values",
    )
    numeric_values = {key: _finite(value, key) for key, value in values.items()}
    sign = 1.0 if direction == "maximize" else -1.0
    gross = numeric_values["closed_loop_gross"]
    cost = numeric_values["expected_sensing_cost"]
    net = numeric_values["closed_loop_net"]
    baseline = numeric_values["no_information"]
    identities = [
        (net, gross - sign * cost, "closed-loop gross/cost/net"),
        (
            numeric_values["gross_information_value"],
            sign * (gross - baseline),
            "gross information value",
        ),
        (
            numeric_values["net_information_value"],
            max(0.0, sign * (net - baseline)),
            "net information value",
        ),
        (
            numeric_values["nonmyopic_information_value"],
            numeric_values["net_information_value"],
            "nonmyopic information value",
        ),
        (
            numeric_values["nonmyopic_minus_myopic"],
            numeric_values["nonmyopic_information_value"]
            - numeric_values["myopic_information_value"],
            "myopic/nonmyopic difference",
        ),
        (
            numeric_values["partial_observability_regret"],
            sign * (numeric_values["fully_observed_value"] - gross),
            "partial-observability regret",
        ),
    ]
    for actual, expected, name in identities:
        if not math.isclose(actual, expected, abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError(f"{name} identity failed")
    if (
        cost < 0.0
        or min(
            numeric_values["net_information_value"],
            numeric_values["myopic_information_value"],
            numeric_values["nonmyopic_information_value"],
            numeric_values["partial_observability_regret"],
        )
        < -1e-8
    ):
        raise ValueError("result costs, values and regret must be nonnegative")

    tree = _require_mapping(result["policy_tree"], "policy_tree")
    root_belief = _require_mapping(tree.get("belief"), "policy_tree.belief")
    state_ids = set(root_belief)
    _validate_policy_tree(tree, state_ids)
    tree_martingale_residual = _policy_martingale_residual(tree, state_ids)
    horizon_curve = _require_list(result["value_by_horizon"], "value_by_horizon")
    if len(horizon_curve) != horizon:
        raise ValueError("value-by-horizon must contain every horizon")
    for index, raw_record in enumerate(horizon_curve, start=1):
        record = _strict_keys(
            raw_record,
            {"horizon", "closed_loop_net", "no_information", "net_information_value"},
            "horizon value",
        )
        if record["horizon"] != index:
            raise ValueError("value-by-horizon records must be consecutive")
        record_net = _finite(record["closed_loop_net"], "horizon closed-loop net")
        record_baseline = _finite(record["no_information"], "horizon baseline")
        record_value = _finite(record["net_information_value"], "horizon value")
        if not math.isclose(
            record_value,
            sign * (record_net - record_baseline),
            abs_tol=1e-8,
            rel_tol=1e-8,
        ):
            raise ValueError("horizon information-value identity failed")
    final_horizon = cast("Mapping[str, Any]", horizon_curve[-1])
    if not math.isclose(
        float(final_horizon["net_information_value"]),
        numeric_values["net_information_value"],
        abs_tol=1e-8,
        rel_tol=1e-8,
    ):
        raise ValueError("final horizon must match the reported information value")

    conditional = _require_list(result["conditional_sensing_values"], "conditional")
    control_ids: set[str] = set()
    for raw_record in conditional:
        record = _strict_keys(
            raw_record, {"control_action_id", "sensors"}, "conditional control"
        )
        control_id = record["control_action_id"]
        if (
            not isinstance(control_id, str)
            or not control_id
            or control_id in control_ids
        ):
            raise ValueError("conditional control IDs must be unique non-empty strings")
        control_ids.add(control_id)
        sensor_ids: set[str] = set()
        for raw_sensor in _require_list(record["sensors"], "conditional sensors"):
            sensor = _strict_keys(
                raw_sensor,
                {
                    "sensor_id",
                    "gross_value",
                    "sensing_cost",
                    "net_value",
                    "net_increment_vs_null",
                },
                "conditional sensor",
            )
            sensor_id = sensor["sensor_id"]
            if (
                not isinstance(sensor_id, str)
                or not sensor_id
                or sensor_id in sensor_ids
            ):
                raise ValueError("conditional sensor IDs must be unique strings")
            sensor_ids.add(sensor_id)
            conditional_gross = _finite(sensor["gross_value"], "conditional gross")
            conditional_cost = _finite(sensor["sensing_cost"], "conditional cost")
            conditional_net = _finite(sensor["net_value"], "conditional net")
            _ = _finite(sensor["net_increment_vs_null"], "conditional increment")
            if conditional_cost < 0.0 or not math.isclose(
                conditional_net,
                conditional_gross - conditional_cost,
                abs_tol=1e-8,
                rel_tol=1e-8,
            ):
                raise ValueError("conditional sensing gross/cost/net identity failed")
        sensors = cast("list[Mapping[str, Any]]", record["sensors"])
        null_candidates = [
            float(sensor["net_value"])
            for sensor in sensors
            if math.isclose(float(sensor["sensing_cost"]), 0.0, abs_tol=1e-8)
            and math.isclose(float(sensor["net_increment_vs_null"]), 0.0, abs_tol=1e-8)
        ]
        if not null_candidates or not all(
            any(
                math.isclose(
                    float(sensor["net_increment_vs_null"]),
                    float(sensor["net_value"]) - null_net,
                    abs_tol=1e-8,
                    rel_tol=1e-8,
                )
                for null_net in null_candidates
            )
            for sensor in sensors
        ):
            raise ValueError("conditional sensor increments must use a null comparator")

    stopping = _strict_keys(result["stopping"], {"kind", "reason", "stage"}, "stopping")
    if stopping != {
        "kind": "fixed_horizon",
        "reason": "horizon_reached",
        "stage": horizon,
    }:
        raise ValueError("stopping result must identify the fixed horizon")
    bounds = _strict_keys(
        result["approximation_bounds"], {"lower", "upper", "gap"}, "bounds"
    )
    lower, upper, gap = (
        _finite(bounds[key], f"bound {key}") for key in ("lower", "upper", "gap")
    )
    if not (
        math.isclose(lower, numeric_values["net_information_value"], abs_tol=1e-8)
        and math.isclose(upper, lower, abs_tol=1e-8)
        and math.isclose(gap, 0.0, abs_tol=1e-8)
    ):
        raise ValueError("exact approximation bounds must have zero gap")
    assurance = _strict_keys(
        result["assurance"],
        {
            "solver",
            "estimator",
            "exact_enumeration",
            "approximation_used",
            "posterior_martingale_verified",
            "posterior_martingale_max_residual",
            "null_sensor_reduction_verified",
            "no_information_reduction_verified",
            "complete_ties_reported",
            "action_dependent_transition",
            "action_dependent_learning",
            "usable_downstream_learning_response",
            "dual_control_diagnostic",
            "unique_additive_dual_control_value_claimed",
            "estimated_bellman_expansions",
            "exact_enumeration_budget",
        },
        "assurance",
    )
    boolean_fields = {
        "exact_enumeration",
        "approximation_used",
        "posterior_martingale_verified",
        "null_sensor_reduction_verified",
        "no_information_reduction_verified",
        "complete_ties_reported",
        "action_dependent_transition",
        "action_dependent_learning",
        "usable_downstream_learning_response",
        "dual_control_diagnostic",
        "unique_additive_dual_control_value_claimed",
    }
    if any(type(assurance[field]) is not bool for field in boolean_fields):
        raise TypeError("assurance flags must be booleans")
    if (
        assurance["solver"] != "exact_finite_horizon_bellman_enumeration"
        or assurance["estimator"] != "deterministic_exact"
        or assurance["exact_enumeration"] is not True
        or assurance["approximation_used"] is not False
        or assurance["unique_additive_dual_control_value_claimed"] is not False
        or assurance["dual_control_diagnostic"]
        != assurance["action_dependent_learning"]
    ):
        raise ValueError("assurance constants or dual-control boundary are invalid")
    estimate = assurance["estimated_bellman_expansions"]
    budget = assurance["exact_enumeration_budget"]
    if (
        isinstance(estimate, bool)
        or not isinstance(estimate, int)
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or estimate < 1
        or estimate > budget
    ):
        raise ValueError("exact-enumeration estimate must fit the declared budget")
    reported_martingale_residual = _finite(
        assurance["posterior_martingale_max_residual"], "martingale residual"
    )
    if not math.isclose(
        reported_martingale_residual,
        tree_martingale_residual,
        abs_tol=1e-10,
        rel_tol=1e-10,
    ) or (
        assurance["posterior_martingale_verified"] is True
        and reported_martingale_residual > 1e-6
    ):
        raise ValueError(
            "posterior-martingale assurance does not match the policy tree"
        )
    language = _strict_keys(
        result["language_dispositions"],
        {"python", "rust", "r", "julia", "mojo"},
        "language dispositions",
    )
    if language != {
        "python": "executable-experimental",
        "rust": "unsupported",
        "r": "unsupported",
        "julia": "unsupported",
        "mojo": "external-boundary",
    }:
        raise ValueError("language dispositions must match experimental evidence")
    limitations = _require_list(result["limitations"], "limitations")
    if not all(isinstance(item, str) and item for item in limitations):
        raise ValueError("limitations must be non-empty strings")


__all__ = [
    "BeliefStateInformationResult",
    "belief_state_information_value",
    "validate_belief_state_information_result",
]
