#!/usr/bin/env python3
"""Measure VOIAGE's higher-dimensional EVPI reduction and retain all decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import cast

import numpy as np

from voiage.c15_oracles import numpy_evpi
from voiage.c15_performance import (
    PerformanceRegressionError,
    current_runner,
    performance_config_digest,
    performance_ratchet,
)

WORKLOAD = "voiage-c15-evpi-4096x4-f64-v1"


def measure(repeats: int, iterations: int) -> list[float]:
    """Collect repeated elapsed-time samples for the fixed C15 workload."""
    if repeats < 5 or iterations <= 0:
        raise ValueError("repeats must be at least five and iterations positive")
    values = np.linspace(-5000, 5000, 4096 * 4, dtype=np.float64).reshape(4096, 4)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        checksum = sum(numpy_evpi(values) for _ in range(iterations))
        if not np.isfinite(checksum):
            raise RuntimeError("performance workload produced a non-finite result")
        samples.append(time.perf_counter() - started)
    return samples


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


def _baseline(path: Path, parameters: dict[str, object]) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("performance baseline must be an object")
    baseline = cast("dict[str, object]", value)
    if (
        baseline.get("workload_id") != WORKLOAD
        or baseline.get("parameters") != parameters
    ):
        raise ValueError("performance baseline workload identity mismatch")
    return baseline


def main() -> int:
    """Run the C15 performance ratchet and retain pass or failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters: dict[str, object] = {
        "repeats": args.repeats,
        "iterations": args.iterations,
        "rows": 4096,
        "strategies": 4,
        "dtype": "float64",
    }
    try:
        baseline = _baseline(args.baseline, parameters)
        report = performance_ratchet(
            measure(args.repeats, args.iterations),
            baseline=baseline,
            runner=current_runner(),
            config_digest=performance_config_digest(
                WORKLOAD, parameters, args.confidence
            ),
            confidence=args.confidence,
        )
    except (
        OSError,
        PerformanceRegressionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _write(
            args.output,
            {
                "schema_version": "1.0.0",
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "baseline": str(args.baseline),
                "parameters": parameters,
            },
        )
        return 2
    _write(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
