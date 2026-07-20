#!/usr/bin/env python3
"""Smoke-validate the core API fixture manifest and artifact layout."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_core_api_contract as validator
from voiage import (
    DecisionOption,
    ParameterSet,
    TrialDesign,
    ValueArray,
    enbs,
    evpi,
    evppi,
    evsi,
)
from voiage import (
    __version__ as voiage_version,
)
from voiage.methods.ceaf import calculate_ceaf
from voiage.methods.dominance import calculate_dominance

DEFAULT_FIXTURE_ROOT = validator.FIXTURE_ROOT.resolve()
STABLE_METHODS = {"evpi", "evppi", "evsi", "enbs", "ceaf", "dominance"}
REQUIRED_CLASSIFICATIONS = {"normal", "edge", "invalid"}


def _constant_model(specification: dict[str, object]):
    values = np.asarray(specification["values"], dtype=float)
    names = [str(name) for name in specification["strategy_names"]]

    def model(parameters: ParameterSet) -> ValueArray:
        return ValueArray.from_numpy(np.tile(values, (parameters.n_samples, 1)), names)

    return model


def _execute_method(method: str, inputs: dict[str, object]) -> object:
    if method == "evpi":
        return evpi(np.asarray(inputs["net_benefit"], dtype=float))
    if method == "evppi":
        parameters = {
            str(name): np.asarray(values, dtype=float)
            for name, values in dict(inputs["parameters"]).items()
        }
        return evppi(
            np.asarray(inputs["net_benefit"], dtype=float),
            ParameterSet.from_numpy_or_dict(parameters),
            [str(name) for name in inputs["parameters_of_interest"]],
        )
    if method == "evsi":
        model_spec = dict(inputs["model"])
        prior = ParameterSet.from_numpy_or_dict(
            {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(inputs["prior"]).items()
            }
        )
        design = TrialDesign(
            arms=[
                DecisionOption(**dict(arm))
                for arm in dict(inputs["trial_design"])["arms"]
            ]
        )
        return evsi(
            _constant_model(model_spec),
            prior,
            design,
            method=str(inputs["estimator"]),
        )
    if method == "enbs":
        return enbs(float(inputs["evsi_result"]), float(inputs["research_cost"]))
    if method == "ceaf":
        values = np.asarray(inputs["net_benefit"], dtype=float)
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies", "threshold"), values)},
            coords={
                "strategy": ("n_strategies", inputs["strategy_names"]),
                "threshold": np.arange(values.shape[2]),
            },
        )
        result = calculate_ceaf(
            ValueArray.from_dataset(dataset),
            inputs["wtp_thresholds"],
            strategy_names=[str(name) for name in inputs["strategy_names"]],
            confidence_level=float(inputs["confidence_level"]),
        )
        return {
            name: getattr(result, name).tolist()
            if hasattr(getattr(result, name), "tolist")
            else getattr(result, name)
            for name in (
                "wtp_thresholds",
                "optimal_strategy_indices",
                "optimal_strategy_names",
                "acceptability_probabilities",
                "probability_lower",
                "probability_upper",
                "expected_net_benefit",
            )
        }
    if method == "dominance":
        costs = np.asarray(inputs["costs"], dtype=float)
        effects = np.asarray(inputs["effects"], dtype=float)
        result = calculate_dominance(
            costs,
            effects,
            [str(name) for name in inputs["strategy_names"]],
        )
        return {
            name: getattr(result, name).tolist()
            if hasattr(getattr(result, name), "tolist")
            else getattr(result, name)
            for name in (
                "frontier_indices",
                "strongly_dominated_indices",
                "extended_dominated_indices",
                "status",
                "incremental_costs",
                "incremental_effects",
                "icers",
            )
        }
    raise validator.ValidationError(f"unsupported method: {method}")


def _assert_result(
    observed: object, expected: object, *, atol: float, rtol: float
) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise AssertionError("result object shape mismatch")
        for key, value in expected.items():
            _assert_result(observed[key], value, atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        if expected and all(isinstance(value, (int, float)) for value in expected):
            np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol)
        elif observed != expected:
            raise AssertionError("result sequence mismatch")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(
            float(observed), float(expected), abs_tol=atol, rel_tol=rtol
        ):
            raise AssertionError("numeric result mismatch")
    elif observed != expected:
        raise AssertionError("result mismatch")


def _assert_json_serializable(observed: object) -> None:
    try:
        json.dumps(observed, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise validator.ValidationError(
            "compatibility outcome must be JSON-serializable"
        ) from exc


def _assert_deterministic_result(first: object, second: object) -> None:
    try:
        _assert_result(second, first, atol=0.0, rtol=0.0)
    except (AssertionError, TypeError, ValueError) as exc:
        raise validator.ValidationError(
            "compatibility outcome must be deterministic"
        ) from exc


def _normalise_error(method: str, exc: Exception) -> dict[str, str]:
    diagnostic_code = getattr(exc, "diagnostic_code", None)
    if diagnostic_code:
        return {"category": "input", "code": str(diagnostic_code)}
    mappings = {
        ("evpi", "DimensionMismatchError"): ("dimension", "shape_mismatch"),
        ("evppi", "InputError"): ("input", "unknown_parameter"),
        ("evsi", "BackendNotAvailableError"): (
            "capability",
            "unsupported_estimator",
        ),
        ("enbs", "InputError"): ("input", "negative_research_cost"),
        ("ceaf", "DimensionMismatchError"): (
            "dimension",
            "threshold_count_mismatch",
        ),
        ("dominance", "DimensionMismatchError"): (
            "dimension",
            "strategy_count_mismatch",
        ),
        ("dominance", "InputError"): ("input", "non_finite_value"),
    }
    if normalized := mappings.get((method, type(exc).__name__)):
        category, code = normalized
        return {"category": category, "code": code}
    return {"category": "unclassified", "code": type(exc).__name__}


def _execute_seeded(method: str, inputs: dict[str, object], *, seed: int) -> object:
    """Execute one case from a freshly applied deterministic NumPy seed."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return _execute_method(method, inputs)
    finally:
        np.random.set_state(state)


def _result_provenance(method: str, *, seed: int) -> dict[str, object]:
    return {
        "voiage_version": voiage_version,
        "core_version": f"python-reference:{voiage_version}",
        "method": method,
        "settings": {"seed": seed},
    }


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise validator.ValidationError(
            "compatibility outcome must be deterministically JSON-serializable"
        ) from exc


def _read_tolerance(expected: dict[str, object], name: str) -> float:
    value = expected.get(name, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise validator.ValidationError(f"{name} must be a finite non-negative number")
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise validator.ValidationError(f"{name} must be a finite non-negative number")
    return tolerance


def _validate_case_index(cases: object) -> list[dict[str, object]]:
    if not isinstance(cases, list) or not cases:
        raise validator.ValidationError("compatibility cases must be a non-empty list")
    typed_cases: list[dict[str, object]] = []
    ids: set[str] = set()
    coverage: set[tuple[str, str]] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise validator.ValidationError("compatibility case must be an object")
        case_id = case.get("case_id")
        method = case.get("method")
        classification = case.get("classification")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise validator.ValidationError("case_id must be non-empty and unique")
        if method not in STABLE_METHODS:
            raise validator.ValidationError(f"{case_id}: unsupported stable method")
        if classification not in REQUIRED_CLASSIFICATIONS:
            raise validator.ValidationError(f"{case_id}: invalid classification")
        ids.add(case_id)
        coverage.add((str(method), str(classification)))
        typed_cases.append(case)
    required = {
        (method, classification)
        for method in STABLE_METHODS
        for classification in REQUIRED_CLASSIFICATIONS
    }
    if missing := sorted(required - coverage):
        raise validator.ValidationError(f"compatibility coverage missing: {missing}")
    return typed_cases


def _validate_manifest_metadata(manifest: dict[str, object]) -> None:
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise validator.ValidationError("seed must be a non-negative integer")
    required_provenance = {
        "reference_implementation": "python",
        "execution_mode": "deterministic",
        "catalog": "v1-python-reference",
    }
    if manifest.get("provenance") != required_provenance:
        raise validator.ValidationError("provenance metadata is missing or invalid")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(validator.REPO_ROOT))
    except ValueError:
        return str(path)


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise validator.ValidationError(f"{path}: expected a JSON object")
    return payload


def _resolve_artifact(relative_path: object, *, directory: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.endswith(".json"):
        raise validator.ValidationError("artifact path must be a relative JSON path")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise validator.ValidationError("artifact path must be relative")
    allowed_root = (validator.FIXTURE_ROOT / "compatibility" / directory).resolve()
    resolved = (validator.FIXTURE_ROOT / candidate).resolve()
    if not resolved.is_relative_to(allowed_root):
        raise validator.ValidationError(f"artifact escapes {allowed_root}")
    return resolved


def execute_compatibility_catalog(
    manifest_path: Path | None = None,
) -> list[dict[str, object]]:
    """Execute language-neutral golden cases against the Python reference API."""
    if manifest_path is None:
        manifest_path = validator.FIXTURE_ROOT / "compatibility-manifest.json"
    manifest = _load_object(manifest_path)
    if manifest.get("version") != "v1":
        raise validator.ValidationError(
            f"{manifest_path}: invalid compatibility catalog"
        )
    _validate_manifest_metadata(manifest)
    seed = int(manifest["seed"])
    cases = _validate_case_index(manifest.get("cases"))

    results: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        method = str(case.get("method", ""))
        input_path = _resolve_artifact(case.get("input_artifact"), directory="inputs")
        expected_path = _resolve_artifact(
            case.get("expected_artifact"), directory="expected"
        )
        inputs = _load_object(input_path)
        expected = _load_object(expected_path)

        try:
            observed = _execute_seeded(method, inputs, seed=seed)
        except Exception as exc:
            error = expected.get("error")
            if not isinstance(error, dict):
                raise validator.ValidationError(
                    f"{case_id}: unexpected error: {exc}"
                ) from exc
            normalized = _normalise_error(method, exc)
            expected_error = {
                "category": error.get("category"),
                "code": error.get("code"),
            }
            if normalized != expected_error:
                raise validator.ValidationError(
                    f"{case_id}: error mismatch: expected {expected_error}, observed {normalized}"
                ) from exc
            try:
                _execute_seeded(method, inputs, seed=seed)
            except Exception as repeated_exc:
                repeated_error = _normalise_error(method, repeated_exc)
                if repeated_error != normalized:
                    raise validator.ValidationError(
                        f"{case_id}: error outcome must be deterministic"
                    ) from repeated_exc
            else:
                raise validator.ValidationError(
                    f"{case_id}: error outcome must be deterministic"
                )
            _assert_json_serializable(normalized)
        else:
            if "error" in expected:
                raise validator.ValidationError(f"{case_id}: expected an error")
            target = expected.get("result")
            try:
                _assert_result(
                    observed,
                    target,
                    atol=_read_tolerance(expected, "absolute_tolerance"),
                    rtol=_read_tolerance(expected, "relative_tolerance"),
                )
            except AssertionError as exc:
                raise validator.ValidationError(
                    f"{case_id}: result mismatch: expected {target}, observed {observed}"
                ) from exc
            repeated = _execute_seeded(method, inputs, seed=seed)
            _assert_deterministic_result(observed, repeated)
            _assert_json_serializable(observed)
        provenance = _result_provenance(method, seed=seed)
        serialized_provenance = _canonical_json(provenance)
        if json.loads(serialized_provenance) != provenance:
            raise validator.ValidationError(f"{case_id}: provenance round-trip failed")
        results.append(
            {
                "case_id": case_id,
                "status": "pass",
                "provenance": provenance,
                "serialized_provenance": serialized_provenance,
            }
        )
    return results


def main() -> int:
    """Validate the committed core API fixture catalog layout."""
    validator.validate_fixture_catalog_layout()
    compatibility_manifest = validator.FIXTURE_ROOT / "compatibility-manifest.json"
    if compatibility_manifest.exists():
        results = execute_compatibility_catalog(compatibility_manifest)
    elif validator.FIXTURE_ROOT.resolve() == DEFAULT_FIXTURE_ROOT:
        raise validator.ValidationError("missing compatibility-manifest.json")
    else:
        results = []
    print(f"validated {_display_path(validator.FIXTURE_MANIFEST)}")
    print(f"executed {len(results)} compatibility cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
