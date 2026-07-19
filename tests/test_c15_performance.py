from __future__ import annotations

from copy import deepcopy

import pytest

from voiage.c15_performance import (
    PerformanceRegressionError,
    confidence_interval,
    current_runner,
    performance_config_digest,
    performance_ratchet,
    runner_fingerprint,
)

PARAMETERS = {
    "repeats": 9,
    "iterations": 40,
    "rows": 4096,
    "strategies": 4,
    "dtype": "float64",
}
DIGEST = performance_config_digest("voiage-c15-evpi-4096x4-f64-v1", PARAMETERS, 0.95)
RUNNER = {
    "os": "Linux",
    "architecture": "x86_64",
    "python_series": "3.14",
    "python_implementation": "CPython",
    "ci_runner_os": "Linux",
    "ci_runner_arch": "x86_64",
    "ci_image_os": "ubuntu24",
}


def _baseline(limit: float = 1.0) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "measurement": {"config_digest": DIGEST, "confidence": 0.95},
        "source": {"commit": "a" * 40, "evidence_kind": "test"},
        "cohorts": {
            "Linux": {
                "approved_runner_identity": RUNNER,
                "approved_runner_fingerprint": runner_fingerprint(RUNNER),
                "maximum_upper_seconds": limit,
            }
        },
    }


def test_bootstrap_interval_is_deterministic_and_bounded() -> None:
    samples = [0.09, 0.10, 0.11, 0.10, 0.10]
    interval = confidence_interval(samples)
    assert interval == confidence_interval(samples)
    assert interval.lower <= interval.mean <= interval.upper
    assert interval.resamples == 10_000


@pytest.mark.parametrize(
    "samples", [[], [1.0], [1.0] * 4, [1.0, 1.0, 1.0, 1.0, float("nan")]]
)
def test_bootstrap_rejects_incomplete_or_nonfinite_samples(
    samples: list[float],
) -> None:
    with pytest.raises(ValueError):
        confidence_interval(samples)


@pytest.mark.parametrize(
    ("confidence", "resamples"), [(0.0, 10_000), (1.0, 10_000), (0.95, 999)]
)
def test_bootstrap_rejects_invalid_configuration(
    confidence: float, resamples: int
) -> None:
    with pytest.raises(ValueError):
        confidence_interval([0.1] * 5, confidence=confidence, resamples=resamples)


def test_runner_identity_and_config_require_exact_nonempty_fields() -> None:
    assert current_runner()["python_series"] == "3.14"
    with pytest.raises(ValueError, match="exact"):
        runner_fingerprint({"os": "Linux"})
    with pytest.raises(ValueError, match="configuration"):
        performance_config_digest("", {}, 0.95)


def test_performance_ratchet_binds_runner_config_source_and_budget() -> None:
    report = performance_ratchet(
        [0.09, 0.10, 0.11, 0.10, 0.10],
        baseline=_baseline(),
        runner=RUNNER,
        config_digest=DIGEST,
    )
    assert report["passed"] is True
    assert report["runner_fingerprint"] == runner_fingerprint(RUNNER)
    assert report["source"] == {"commit": "a" * 40, "evidence_kind": "test"}


def test_performance_ratchet_rejects_regression_and_cohort_drift() -> None:
    with pytest.raises(PerformanceRegressionError):
        performance_ratchet(
            [0.9, 1.0, 1.1, 1.0, 1.0],
            baseline=_baseline(0.5),
            runner=RUNNER,
            config_digest=DIGEST,
        )
    drifted = deepcopy(RUNNER)
    drifted["ci_image_os"] = "ubuntu26"
    with pytest.raises(ValueError, match="approved cohort"):
        performance_ratchet(
            [0.1] * 5,
            baseline=_baseline(),
            runner=drifted,
            config_digest=DIGEST,
        )
    with pytest.raises(ValueError, match="configuration"):
        performance_ratchet(
            [0.1] * 5,
            baseline=_baseline(),
            runner=RUNNER,
            config_digest="0" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version="2.0.0"),
        lambda value: value.pop("measurement"),
        lambda value: value["cohorts"].pop("Linux"),
        lambda value: value["cohorts"]["Linux"].pop("approved_runner_identity"),
        lambda value: value["cohorts"]["Linux"].update(
            approved_runner_fingerprint="0" * 64
        ),
        lambda value: value["cohorts"]["Linux"].update(maximum_upper_seconds=0),
    ],
)
def test_performance_ratchet_rejects_malformed_retained_evidence(mutation) -> None:
    baseline = _baseline()
    mutation(baseline)
    with pytest.raises((TypeError, ValueError)):
        performance_ratchet(
            [0.1] * 5,
            baseline=baseline,
            runner=RUNNER,
            config_digest=DIGEST,
        )
