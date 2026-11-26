"""Compare frozen pre-container source and the candidate on one Python runtime."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from types import ModuleType

import numpy as np

from voiage.schema import ValueArray

HERE = Path(__file__).resolve().parent
REPEATS = 100
BASELINE_SHA256 = "6fd5b102b04ff7b2e2b649e38beb46173e8b2d404d6412c1ad6ab4cf9432e7bb"


def main() -> int:
    """Emit a matched-environment receipt without relaxing the retained bounds."""
    source = (HERE / "benchmark-baseline-schema-2a6e76b7.txt").read_bytes()
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
        raise ValueError("frozen baseline source digest changed")
    baseline = ModuleType("_voiage_pre_container_benchmark")
    sys.modules[baseline.__name__] = baseline
    # Execute only the committed, SHA256-verified historical project source.
    exec(compile(source, "baseline-schema-2a6e76b7", "exec"), baseline.__dict__)  # noqa: S102
    rng = np.random.default_rng(20261026)
    values = rng.standard_normal((20_000, 5))
    perspectives = rng.standard_normal((5_000, 5, 4))
    names = [f"S{i}" for i in range(5)]
    old = baseline.ValueArray.from_numpy(values, names)
    new = ValueArray.from_numpy(values, names)
    np.testing.assert_array_equal(old.numpy_values, new.numpy_values)
    np.testing.assert_array_equal(
        baseline.ValueArray.from_numpy_perspectives(perspectives, names).numpy_values,
        ValueArray.from_numpy_perspectives(perspectives, names).numpy_values,
    )
    pairs = {
        "construction_2d": (
            lambda: baseline.ValueArray.from_numpy(values, names),
            lambda: ValueArray.from_numpy(values, names),
        ),
        "construction_3d": (
            lambda: baseline.ValueArray.from_numpy_perspectives(perspectives, names),
            lambda: ValueArray.from_numpy_perspectives(perspectives, names),
        ),
        "to_numpy_extraction": (lambda: old.numpy_values, lambda: new.numpy_values),
        "wide_to_long_conversion": (
            lambda: old.to_dataset()["net_benefit"].to_dataframe(),  # noqa: PLW0108 - include conversion in timing
            lambda: new.to_polars().unpivot(on=names),
        ),
    }
    timings = {}
    for name, operations in pairs.items():
        samples = [[], []]
        for operation in operations:
            operation()
        for index in range(REPEATS):
            for side in (0, 1) if index % 2 == 0 else (1, 0):
                start = time.perf_counter_ns()
                operations[side]()
                samples[side].append((time.perf_counter_ns() - start) / 1_000_000)
        timings[name] = dict(
            zip(("before_ms", "after_ms"), map(statistics.median, samples), strict=True)
        )
    accepted = (
        all(
            timings[name]["after_ms"] <= 1.5 * timings[name]["before_ms"]
            for name in ("construction_2d", "construction_3d")
        )
        and timings["to_numpy_extraction"]["after_ms"] <= 0.5
        and timings["wide_to_long_conversion"]["after_ms"]
        <= timings["wide_to_long_conversion"]["before_ms"]
    )
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "python": platform.python_version(),
                "packages": {
                    name: version(name)
                    for name in ("numpy", "xarray", "pandas", "polars")
                },
                "baseline_source_sha256": hashlib.sha256(source).hexdigest(),
                "candidate_schema_sha256": hashlib.sha256(
                    (HERE.parents[2] / "voiage/schema.py").read_bytes()
                ).hexdigest(),
                "fixture": {
                    "shape_2d": list(values.shape),
                    "shape_3d": list(perspectives.shape),
                    "seed": 20261026,
                    "repeats": REPEATS,
                    "order": "alternating paired",
                },
                "timings": timings,
                "accepted": accepted,
                "bounds": {
                    "construction_ratio_maximum": 1.5,
                    "extraction_maximum_ms": 0.5,
                    "long_form_must_not_regress": True,
                },
            },
            indent=2,
        )
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
