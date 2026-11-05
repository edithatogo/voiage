# SPDX-License-Identifier: Apache-2.0
"""Project Innovate's parameter fixture onto the native Bass API."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys


def prepare(source: Path, destination: Path) -> None:
    rows: dict[tuple[str, int], dict[str, float]] = {}
    data = source.read_bytes()
    if (
        hashlib.sha256(data).hexdigest()
        != "411cf641d44260a35f5359550cf36459d64ea52e2d659368f289df4231c5257d"
    ):
        raise ValueError("Innovate parameter fixture digest mismatch")
    with io.StringIO(data.decode("utf-8")) as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["draw_id"]))
            params = rows.setdefault(key, {})
            if row["parameter_name"] in params:
                raise ValueError(f"duplicate parameter {key}: {row['parameter_name']}")
            params[row["parameter_name"]] = float(row["value"])
    if len(rows) != 6 or any(
        set(params) != {"p", "q", "m"} for params in rows.values()
    ):
        raise ValueError("expected six complete Bass scenarios")
    payload = [
        {"scenario_id": scenario, "draw_id": draw, "parameters": params}
        for (scenario, draw), params in sorted(rows.items())
    ]
    with destination.open("x") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
