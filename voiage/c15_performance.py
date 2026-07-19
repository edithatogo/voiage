"""Runner-cohort-bound deterministic bootstrap performance assurance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
import platform
import random
import statistics

BOOTSTRAP_METHOD = "deterministic_percentile_bootstrap"


class PerformanceRegressionError(RuntimeError):
    """Raised when a confidence upper bound exceeds its cohort budget."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Summary and percentile bounds for repeated timing samples."""

    method: str
    confidence: float
    samples: int
    resamples: int
    mean: float
    standard_deviation: float
    lower: float
    upper: float


def _percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def confidence_interval(
    samples: list[float], *, confidence: float = 0.95, resamples: int = 10_000
) -> ConfidenceInterval:
    """Return a reproducible percentile-bootstrap interval."""
    if len(samples) < 5:
        raise ValueError("at least five samples are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples < 1_000:
        raise ValueError("bootstrap resamples must be at least 1000")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("performance samples must be finite and non-negative")
    generator = random.Random(0xC15)  # nosec B311  # noqa: S311
    size = len(samples)
    means = sorted(
        statistics.fmean(samples[generator.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    return ConfidenceInterval(
        method=BOOTSTRAP_METHOD,
        confidence=confidence,
        samples=size,
        resamples=resamples,
        mean=statistics.fmean(samples),
        standard_deviation=statistics.stdev(samples),
        lower=_percentile(means, tail),
        upper=_percentile(means, 1 - tail),
    )


def _architecture(value: object) -> str:
    normalized = str(value).strip().casefold().replace("-", "_")
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(
        normalized, normalized
    )


def _environment_value(name: str) -> str:
    return os.environ.get(name, "local")


def current_runner() -> dict[str, str]:
    """Return stable fields defining a comparable hosted-runner cohort."""
    return {
        "os": platform.system(),
        "architecture": _architecture(platform.machine()),
        "python_series": ".".join(platform.python_version().split(".")[:2]),
        "python_implementation": platform.python_implementation(),
        "ci_runner_os": os.environ.get("RUNNER_OS", "local"),
        "ci_runner_arch": _architecture(
            os.environ.get("RUNNER_ARCH", platform.machine())
        ),
        "ci_image_os": _environment_value("ImageOS").casefold(),
    }


def runner_fingerprint(runner: Mapping[str, object]) -> str:
    """Hash an exact canonical runner cohort identity."""
    required = {
        "os",
        "architecture",
        "python_series",
        "python_implementation",
        "ci_runner_os",
        "ci_runner_arch",
        "ci_image_os",
    }
    if set(runner) != required or any(not str(runner[key]).strip() for key in required):
        raise ValueError(
            "runner identity must contain the exact non-empty cohort fields"
        )
    return sha256(
        json.dumps(dict(runner), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def performance_config_digest(
    workload_id: str, parameters: Mapping[str, object], confidence: float
) -> str:
    """Bind workload parameters and interval method to baseline evidence."""
    if not workload_id.strip() or not parameters:
        raise ValueError("performance configuration requires workload identity")
    value = {
        "workload_id": workload_id,
        "parameters": dict(parameters),
        "interval_method": BOOTSTRAP_METHOD,
        "confidence": confidence,
        "resamples": 10_000,
        "minimum_samples": 5,
    }
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def performance_ratchet(
    samples: list[float],
    *,
    baseline: Mapping[str, object],
    runner: Mapping[str, object],
    config_digest: str,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Enforce the exact runner cohort, config digest, and bootstrap upper budget."""
    if baseline.get("schema_version") != "1.0.0":
        raise ValueError("unsupported performance baseline schema")
    measurement = baseline.get("measurement")
    cohorts = baseline.get("cohorts")
    source = baseline.get("source")
    if (
        not isinstance(measurement, Mapping)
        or not isinstance(cohorts, Mapping)
        or not isinstance(source, Mapping)
    ):
        raise TypeError("performance baseline is missing retained evidence")
    if (
        measurement.get("config_digest") != config_digest
        or measurement.get("confidence") != confidence
    ):
        raise ValueError("performance measurement configuration mismatch")
    os_name = str(runner.get("os", ""))
    cohort = cohorts.get(os_name)
    if not isinstance(cohort, Mapping):
        raise TypeError(f"performance baseline has no {os_name!r} cohort")
    approved = cohort.get("approved_runner_identity")
    fingerprint = cohort.get("approved_runner_fingerprint")
    if not isinstance(approved, Mapping) or not isinstance(fingerprint, str):
        raise TypeError("performance cohort lacks an approved runner identity")
    approved_dict = {str(key): str(value) for key, value in approved.items()}
    current_dict = {str(key): str(value) for key, value in runner.items()}
    if runner_fingerprint(approved_dict) != fingerprint:
        raise ValueError("approved runner fingerprint is inconsistent")
    if current_dict != approved_dict or runner_fingerprint(current_dict) != fingerprint:
        raise ValueError("current runner does not match the approved cohort")
    limit = cohort.get("maximum_upper_seconds")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not math.isfinite(float(limit))
        or float(limit) <= 0
    ):
        raise ValueError("performance cohort budget must be positive and finite")
    interval = confidence_interval(samples, confidence=confidence)
    if interval.upper > float(limit):
        raise PerformanceRegressionError(
            f"upper confidence bound {interval.upper:.9f}s exceeds budget {float(limit):.9f}s"
        )
    return {
        "schema_version": "1.0.0",
        "passed": True,
        "runner": current_dict,
        "runner_fingerprint": fingerprint,
        "interval": asdict(interval),
        "maximum_upper_seconds": float(limit),
        "samples": samples,
        "source": dict(source),
        "measurement": dict(measurement),
    }


__all__ = [
    "ConfidenceInterval",
    "PerformanceRegressionError",
    "confidence_interval",
    "current_runner",
    "performance_config_digest",
    "performance_ratchet",
    "runner_fingerprint",
]
