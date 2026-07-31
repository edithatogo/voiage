"""Independent references for estimation-focused variance VOI."""

# pyright: reportAny=false, reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import pvariance

import pytest

REFERENCE_PATH = (
    Path(__file__).parent / "data" / "estimation_variance_reference_v1.json"
)


def _cases() -> dict[str, dict[str, object]]:
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    return {case["case_id"]: case for case in payload["cases"]}


def test_evppi_var_discrete_reference_is_independently_enumerable() -> None:
    case = _cases()["evppi-var-discrete-partial"]
    samples = [float(value) for value in case["target_samples"]]
    groups = [str(value) for value in case["conditioning_groups"]]
    group_values = {
        group: [value for value, label in zip(samples, groups, strict=True) if label == group]
        for group in set(groups)
    }

    prior_variance = pvariance(samples)
    expected_posterior_variance = sum(
        len(values) / len(samples) * pvariance(values)
        for values in group_values.values()
    )
    reduction = prior_variance - expected_posterior_variance

    assert prior_variance == pytest.approx(case["prior_variance"])
    assert expected_posterior_variance == pytest.approx(
        case["expected_posterior_variance"]
    )
    assert reduction == pytest.approx(case["variance_reduction"])


def test_evsi_var_normal_normal_reference_matches_closed_form() -> None:
    case = _cases()["evsi-var-normal-normal-n6"]
    prior_variance = float(case["prior_variance"])
    sampling_variance = float(case["sampling_variance"])
    sample_size = int(case["sample_size"])

    posterior_variance = 1.0 / (
        (1.0 / prior_variance) + (sample_size / sampling_variance)
    )
    reduction = prior_variance - posterior_variance

    assert posterior_variance == pytest.approx(case["expected_posterior_variance"])
    assert reduction == pytest.approx(case["variance_reduction"])


def test_evsi_var_binary_reference_enumerates_posterior_outcomes() -> None:
    case = _cases()["evsi-var-binary-enumeration"]
    prior_probability = float(case["prior_probability"])
    accuracy = float(case["observation_accuracy"])
    expected_posterior_variance = 0.0
    for observed in (0, 1):
        likelihood_if_one = accuracy if observed == 1 else 1.0 - accuracy
        likelihood_if_zero = 1.0 - accuracy if observed == 1 else accuracy
        probability_observed = (
            likelihood_if_one * prior_probability
            + likelihood_if_zero * (1.0 - prior_probability)
        )
        posterior_probability = (
            likelihood_if_one * prior_probability / probability_observed
        )
        expected_posterior_variance += probability_observed * (
            posterior_probability * (1.0 - posterior_probability)
        )

    prior_variance = prior_probability * (1.0 - prior_probability)
    reduction = prior_variance - expected_posterior_variance

    assert math.isclose(prior_variance, float(case["prior_variance"]))
    assert expected_posterior_variance == pytest.approx(
        case["expected_posterior_variance"]
    )
    assert reduction == pytest.approx(case["variance_reduction"])
