# SPDX-License-Identifier: Apache-2.0
"""Validate fresh Innovate Rust Bass trajectories and evaluate an application payoff."""

import argparse
from decimal import Decimal, localcontext
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from voiage.methods.basic import evpi


def evaluate(input_path: Path, output_path: Path) -> dict[str, object]:
    """Check every keyed native prediction against an independent Decimal oracle."""
    source = input_path.read_bytes()
    if (
        sha256(source).hexdigest()
        != "962ba817a812dc9a330170bfe47175efcad77d7463eab7532a14cb09b4bf1986"
    ):
        raise ValueError("Native Innovate witness input digest mismatch")
    inputs = json.loads(source)
    outputs = json.loads(output_path.read_text())
    predictions = {(r["scenario_id"], r["draw_id"]): r["prediction"] for r in outputs}
    assert len(outputs) == len(predictions) == len(inputs) == 6
    reference = {}
    max_abs_error = Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 50
        for row in inputs:
            key = (row["scenario_id"], row["draw_id"])
            p, q, m = [
                Decimal(str(row["parameters"][name])) for name in ("p", "q", "m")
            ]
            points = predictions[key]
            assert len(points) == 4
            for t, actual in enumerate(points):
                e = (-(p + q) * t).exp()
                expected = m * (1 - e) / (1 + (q / p) * e)
                max_abs_error = max(
                    max_abs_error, abs(Decimal.from_float(actual) - expected)
                )
            reference[key] = expected
        assert max_abs_error < Decimal("1e-10")
        # The consumer assigns a synthetic 35-person-equivalent penalty to acceleration.
        draws = [1, 2, 3]
        exact = [
            (reference["baseline_access", i], reference["accelerated_access", i] - 35)
            for i in draws
        ]
        reference_evpi = sum(map(max, exact)) / 3 - max(
            sum(col) / 3 for col in zip(*exact, strict=True)
        )
        nb = np.array(
            [
                (
                    predictions["baseline_access", i][-1],
                    predictions["accelerated_access", i][-1] - 35,
                )
                for i in draws
            ],
            dtype=np.float64,
        )
        values = [float(evpi(v)) for v in (nb, nb[::-1], nb[:, ::-1])]
        assert reference_evpi > 0
        assert all(
            abs(Decimal.from_float(v) - reference_evpi) < Decimal("1e-10")
            for v in values
        )
    return {
        "python_consumer": version("voiage"),
        "producer": "innovate-rs 0.5.0",
        "rust_kernel": "KernelBinding::predict_model_native",
        "draw_scenario_pairs": len(predictions),
        "trajectory_points_checked": 24,
        "max_prediction_absolute_error": str(max_abs_error),
        "application_cost": 35,
        "decimal_evpi": str(reference_evpi),
        "observed_evpi": values,
        "input_sha256": sha256(input_path.read_bytes()).hexdigest(),
        "output_sha256": sha256(output_path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input_path, args.output_path), indent=2))
