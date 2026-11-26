#!/usr/bin/env python3
"""Numerical parity reference harness for the container swap (AC4/AC5).

Runs the same public-API sequence against whatever ``voiage.schema`` is on the
path and prints a JSON digest of every NumPy array produced. Run it once against
the pre-container revision and once against the candidate; the digests must be
bit-identical (``np.array_equal``).

This script imports only ``voiage`` public API plus numpy, so it executes
unchanged on both the xarray container (before) and the Polars container (after).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from voiage.schema import ParameterSet, ValueArray

VALUE_ARRAY_EXAMPLE = {
    "strategy_names": ["Usual care", "Targeted screening"],
    "net_benefit": [
        [1500.0, 1625.0],
        [1490.0, 1650.0],
        [1510.0, 1635.0],
    ],
}
PARAMETER_SET_EXAMPLE = {
    "parameters": {
        "prob_screen_positive": [0.08, 0.1, 0.09],
        "incremental_cost": [120.0, 140.0, 130.0],
        "incremental_effect": [0.02, 0.03, 0.025],
    }
}


def _digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _record(results: dict[str, object], name: str, array: np.ndarray) -> None:
    results[name] = {
        "sha256": _digest(array),
        "shape": list(np.asarray(array).shape),
        "dtype": str(np.asarray(array).dtype),
    }


def main() -> int:
    rng = np.random.default_rng(20261026)
    results: dict[str, object] = {}
    order: dict[str, object] = {}

    # Reference case 1: 2D net benefit, declared column order.
    values_2d = rng.standard_normal((64, 4))
    va2 = ValueArray.from_numpy(values_2d, strategy_names=["A", "B", "C", "D"])
    _record(results, "va_2d_values", va2.numpy_values)
    _record(results, "va_2d_input", values_2d)
    order["va_2d_columns"] = va2.strategy_names

    # Reference case 2: single strategy / single sample edge shapes.
    va_edge = ValueArray.from_numpy(np.array([[3.25]]), strategy_names=["only"])
    _record(results, "va_edge_values", va_edge.numpy_values)

    # Reference case 3: 3D multi-perspective surface.
    values_3d = rng.standard_normal((16, 3, 2))
    va3 = ValueArray.from_numpy_perspectives(
        values_3d, strategy_names=["A", "B", "C"], perspective_names=["payer", "societal"]
    )
    _record(results, "va_3d_values", va3.numpy_values)
    order["va_3d_columns"] = va3.strategy_names
    order["va_3d_perspectives"] = va3.perspective_names

    # Reference case 4: ParameterSet from a 2D array.
    params_2d = rng.standard_normal((64, 3))
    ps2 = ParameterSet.from_numpy_or_dict(params_2d)
    _record(results, "ps_2d_param0", ps2.parameters["param_0"])
    _record(results, "ps_2d_input", params_2d)
    order["ps_2d_names"] = ps2.parameter_names

    # Reference case 5: ParameterSet from a named dict.
    params_dict = {
        "alpha": rng.standard_normal(32),
        "beta": rng.standard_normal(32),
    }
    ps_dict = ParameterSet.from_numpy_or_dict(params_dict)
    _record(results, "ps_dict_alpha", ps_dict.parameters["alpha"])
    order["ps_dict_names"] = ps_dict.parameter_names

    # Reference case 6: the specs/core-api v1 example fixtures.
    va_example = ValueArray.from_numpy(
        np.asarray(VALUE_ARRAY_EXAMPLE["net_benefit"], dtype=np.float64),
        strategy_names=list(VALUE_ARRAY_EXAMPLE["strategy_names"]),
    )
    _record(results, "va_example_values", va_example.numpy_values)
    ps_example = ParameterSet.from_numpy_or_dict(
        {
            key: np.asarray(val, dtype=np.float64)
            for key, val in PARAMETER_SET_EXAMPLE["parameters"].items()
        }
    )
    _record(results, "ps_example_cost", ps_example.parameters["incremental_cost"])
    order["ps_example_names"] = ps_example.parameter_names

    print(json.dumps({"arrays": results, "order": order}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())