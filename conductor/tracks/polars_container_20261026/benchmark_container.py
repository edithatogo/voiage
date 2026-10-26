#!/usr/bin/env python3
"""Container cost benchmark for the Polars swap (AC6).

Measures construction, ``to_numpy`` extraction, and wide-to-long conversion on a
fixed fixture. It detects the container API so the same script runs against the
pre-swap xarray container and the Polars container. Prints median milliseconds
per operation as JSON.

Run:  uv run python benchmark_container.py
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np

from voiage.schema import ValueArray

N_SAMPLES = 20_000
N_STRATEGIES = 5
N_PERSPECTIVES = 4
REPEATS = 100
SEED = 20261026


def _time(operation, repeats: int = REPEATS) -> float:
    """Return the median milliseconds per call over ``repeats`` timed calls."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    rng = np.random.default_rng(SEED)
    values = rng.standard_normal((N_SAMPLES, N_STRATEGIES))
    strategy_names = [f"S{i}" for i in range(N_STRATEGIES)]
    perspective_values = rng.standard_normal(
        (N_SAMPLES // 4, N_STRATEGIES, N_PERSPECTIVES)
    )

    is_polars = hasattr(ValueArray, "from_polars")

    # Construction
    construct = _time(lambda: ValueArray.from_numpy(values, strategy_names))
    construct_persp = _time(
        lambda: ValueArray.from_numpy_perspectives(
            perspective_values, strategy_names
        )
    )

    value_array = ValueArray.from_numpy(values, strategy_names)

    # to_numpy extraction
    extract = _time(lambda: value_array.numpy_values)

    # Wide-to-long conversion (container-native where available)
    if is_polars:
        def long_form() -> None:
            value_array.to_polars().unpivot(
                on=strategy_names,
                variable_name="strategy",
                value_name="net_benefit",
            )
    else:
        def long_form() -> None:
            # xarray's DataArray.to_dataframe already yields the long form.
            value_array.to_dataset()["net_benefit"].to_dataframe()
    long_ms = _time(long_form)

    receipt = {
        "fixture": {
            "n_samples": N_SAMPLES,
            "n_strategies": N_STRATEGIES,
            "n_perspectives": N_PERSPECTIVES,
            "seed": SEED,
            "repeats": REPEATS,
        },
        "container": "polars" if is_polars else "xarray",
        "median_ms": {
            "construction_2d": round(construct, 4),
            "construction_3d": round(construct_persp, 4),
            "to_numpy_extraction": round(extract, 4),
            "wide_to_long_conversion": round(long_ms, 4),
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())