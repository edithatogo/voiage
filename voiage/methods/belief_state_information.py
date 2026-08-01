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
        math.fsum(initial.values()), 1.0, abs_tol=probability_tolerance
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
            likelihood[null_sensor][action][state] != reference for state in states[1:]
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
    if not math.isclose(math.fsum(item[1] for item in branches), 1.0, abs_tol=1e-10):
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


def _adaptive(model: _Model, stage: int, belief: dict[str, float]) -> _Evaluation:
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
                    _adaptive(model, stage + 1, posterior)
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
    return _Evaluation(selected.net, selected.gross, selected.sensing_cost, tree)


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


def _conditional_sensing(model: _Model) -> list[dict[str, Any]]:
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
                    child = _adaptive(model, 1, posterior)
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


def _action_dependent_learning(model: _Model) -> bool:
    for sensor in model.sensors:
        for left in model.actions:
            for right in model.actions:
                if model.likelihood[sensor][left] != model.likelihood[sensor][right]:
                    return True
    return False


def _evaluate(model: _Model) -> dict[str, Any]:
    adaptive = _adaptive(model, 0, model.initial_belief)
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
    action_learning = _action_dependent_learning(model)
    transition_dependent = (
        len(
            {
                json.dumps(model.transition[action], sort_keys=True)
                for action in model.actions
            }
        )
        > 1
    )
    conditional = _conditional_sensing(model)
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
            "posterior_martingale_verified": martingale_residual <= 1e-10,
            "posterior_martingale_max_residual": martingale_residual,
            "null_sensor_reduction_verified": null_reduction,
            "no_information_reduction_verified": net_voi >= -model.absolute_tie,
            "complete_ties_reported": True,
            "action_dependent_transition": transition_dependent,
            "action_dependent_learning": action_learning,
            "dual_control_diagnostic": action_learning or transition_dependent,
            "unique_additive_dual_control_value_claimed": False,
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


__all__ = ["BeliefStateInformationResult", "belief_state_information_value"]
