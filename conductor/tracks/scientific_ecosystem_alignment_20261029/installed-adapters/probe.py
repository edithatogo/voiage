# SPDX-License-Identifier: Apache-2.0
"""Check two explicit application-owned handoffs against installed distributions."""

import argparse
import csv
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import io
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def innovate(root: Path) -> dict[str, object]:
    from voiage.methods.basic import evpi

    # Independently recorded at Innovate commit 83d556ba. Parse only verified bytes.
    expected = {
        "manifest.json": "b61feb8ed857dc1ab5f9b6d18d75091fe5b458c345cdace66338d0ae6afcbc5e",
        "parameter_draws.csv": "411cf641d44260a35f5359550cf36459d64ea52e2d659368f289df4231c5257d",
        "adoption_trajectories.csv": "ba457afdab6a001d5fb5273cf03b20bbf0ac93189a260eaed761a541fd4fb1fb",
    }
    payload = {}
    for name, wanted in expected.items():
        data = (root / name).read_bytes()
        if sha256(data).hexdigest() != wanted:
            raise ValueError(f"Innovate fixture digest mismatch: {name}")
        payload[name] = data
    manifest = json.loads(payload["manifest.json"])
    tables = {}
    for key, spec in manifest["files"].items():
        reader = csv.DictReader(io.StringIO(payload[spec["path"]].decode("utf-8")))
        assert reader.fieldnames == [item["name"] for item in spec["columns"]]
        rows = list(reader)
        ids = [tuple(row[col] for col in spec["primary_key"]) for row in rows]
        assert len(set(ids)) == len(rows) == spec["row_count"]
        tables[key] = rows
    trajectories = tables["adoption_trajectories"]
    parameters = tables["parameter_draws"]
    assert {(x["scenario_id"], x["draw_id"]) for x in parameters} == {
        (x["scenario_id"], x["draw_id"]) for x in trajectories
    }
    # This penalty is an explicit synthetic payoff assumption in adoption units.
    penalty = Fraction(1, 20)
    terminal = {
        (r["scenario_id"], r["draw_id"]): Fraction(r["cumulative_adoption"])
        for r in trajectories
        if r["time"] == "3"
    }

    def payoff(draw):
        return (
            terminal["baseline_access", draw],
            terminal["accelerated_access", draw] - penalty,
        )

    exact = [payoff(d) for d in ("1", "2", "3")]
    reference = sum(map(max, exact)) / 3 - max(
        sum(c) / 3 for c in zip(*exact, strict=True)
    )
    assert reference == Fraction(1, 600)
    matrix = np.array(exact, dtype=np.float64)
    values = [float(evpi(values)) for values in (matrix, matrix[::-1], matrix[:, ::-1])]
    assert all(abs(v - float(reference)) < 1e-14 for v in values)
    return {
        "fixture_revision": "83d556ba4995fd5f5238f4953e9a8af34976f464",
        "files": expected,
        "reference": str(reference),
        "synthetic_penalty": str(penalty),
        "observed": values,
        "rows": {key: len(rows) for key, rows in tables.items()},
    }


def mars() -> dict[str, object]:
    from pymars import EarthRegressor
    from voiage.methods.basic import evppi
    from voiage.schema import ParameterSet

    count = 33
    x = np.linspace(-1.0, 1.0, count)
    # Two competing affine conditional means, with all uncertainty observed.
    nb = np.column_stack((x, -x))
    reference = sum(abs(Fraction(i, 16)) for i in range(-16, 17)) / count
    assert reference == Fraction(17, 33)
    result = float(
        evppi(
            nb,
            ParameterSet.from_numpy_or_dict({"slope": x}),
            ["slope"],
            regression_model=EarthRegressor,
        )
    )
    if not np.isclose(result, float(reference), rtol=1e-10, atol=1e-12):
        raise AssertionError((result, reference))
    return {
        "mars_revision": "d68b54a1c9a5eee40e1b7a724e002d5083ff6d8f",
        "estimator": "pymars.EarthRegressor",
        "mars_version": version("mars-earth"),
        "samples": count,
        "reference": str(reference),
        "estimate": result,
        "estimator_module_in_environment": str(
            sys.modules["pymars"].__file__
        ).startswith(sys.prefix + "/"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["innovate", "mars"])
    parser.add_argument("fixture", type=Path)
    pins = json.loads(Path(__file__).with_name("package-identities.json").read_text())
    parser.add_argument(
        "--voiage-sha", default=pins["candidate_voiage"]["wheel_sha256"]
    )
    args = parser.parse_args()
    verify_wheel = runpy.run_path(str(Path(__file__).with_name("wheel_identity.py")))[
        "verify_wheel"
    ]
    verify_wheel("voiage", "voiage", "2.2.0", args.voiage_sha)
    if args.mode == "mars":
        verify_wheel("mars-earth", "pymars", "1.0.4", pins["mars"]["wheel_sha256"])
    result = innovate(args.fixture) if args.mode == "innovate" else mars()
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "voiage": version("voiage"),
                "voiage_in_environment": str(sys.modules["voiage"].__file__).startswith(
                    sys.prefix + "/"
                ),
                **result,
            },
            indent=2,
        )
    )
