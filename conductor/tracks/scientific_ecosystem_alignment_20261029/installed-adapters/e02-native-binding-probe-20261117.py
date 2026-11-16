"""Exercise an installed Innovate-to-Voiage E02 candidate with synthetic values.

The synthetic application values one terminal adoption as one utility unit and
the alternative at 16 units, with no additional costs. This is a mathematical
test assumption, not a clinical or economic valuation or a validated fit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import innovate
import innovate._native
import polars as pl

import voiage
import voiage._core
import voiage.innovate_binding


def sha256(path: Path) -> str:
    """Return the digest of an input artifact or installed module."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_module(wheel: Path, package: str, module_path: str) -> None:
    """Verify an imported module is installed here and matches its wheel member."""
    path = Path(module_path).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"{path} is outside the selected environment")
    member = f"{package}/{path.name}"
    with ZipFile(wheel) as archive:
        if hashlib.sha256(archive.read(member)).hexdigest() != sha256(path):
            raise ValueError(f"{member} differs from the supplied wheel")


def main() -> None:
    """Run the synthetic cross-package probe and print its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--innovate-wheel", type=Path, required=True)
    parser.add_argument("--voiage-wheel", type=Path, required=True)
    args = parser.parse_args()
    expected_innovate = (
        "4547a4ba432ad63b89cc6f7e661df0ec55b3af53377330a146660ccd86c8440b"
    )
    expected_voiage = "0859daa9571396cf385771e955f7107a58971948289c550442aeb8a44b6591d8"
    if sha256(args.innovate_wheel) != expected_innovate:
        raise ValueError("Innovate wheel digest changed")
    if sha256(args.voiage_wheel) != expected_voiage:
        raise ValueError("Voiage wheel digest changed")
    verify_module(args.innovate_wheel, "innovate", innovate.__file__)
    verify_module(args.innovate_wheel, "innovate", innovate._native.__file__)
    verify_module(args.voiage_wheel, "voiage", voiage.innovate_binding.__file__)
    verify_module(args.voiage_wheel, "voiage", voiage._core.__file__)

    rates = (0.01, 0.08)
    scenarios = [
        {
            "id": f"diffusion-{index}",
            "model_key": "bass",
            "simulation": {"parameters": {"p": p, "q": 0.3, "m": 100.0}},
        }
        for index, p in enumerate(rates)
    ]
    scenarios.append(
        {
            "id": "alternative",
            "model_key": "logistic",
            "simulation": {"parameters": {"L": 32.0, "k": 1.0, "x0": 3.0}},
        }
    )
    response = innovate.execute_scenario_grid(
        pl.DataFrame({"time": [0.0, 3.0]}), scenarios=scenarios
    )
    executions = response["result"]["executions"]
    if [row["id"] for row in executions] != [scenario["id"] for scenario in scenarios]:
        raise ValueError("Innovate scenario identities changed")
    if any(row["error"] is not None for row in executions):
        raise ValueError("Innovate scenario execution failed")
    bass = [row["result"]["values"][-1] for row in executions[:2]]
    alternative = executions[2]["result"]["values"][-1]
    for rate, actual in zip(rates, bass, strict=True):
        decay = math.exp(-(rate + 0.3) * 3.0)
        analytic = 100.0 * (1.0 - decay) / (1.0 + 0.3 / rate * decay)
        if not math.isclose(actual, analytic, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Innovate Bass result disagrees with the analytic oracle")
    if alternative != 16.0 or not (bass[0] < alternative < bass[1]):
        raise ValueError("The synthetic strategies do not switch by draw")
    if sum(bass) / 2 <= alternative:
        raise ValueError("The synthetic prior decision is not diffusion")

    payoffs = pl.DataFrame(
        {
            "draw_id": ["low", "high"],
            "diffusion": bass,
            "alternative": [alternative] * 2,
        }
    )
    ipc = innovate.export_ipc_file(payoffs)
    restored = pl.read_ipc(io.BytesIO(ipc))
    if not restored.equals(payoffs):
        raise ValueError("Arrow IPC changed the labelled synthetic utilities")
    binding = {
        "schema_version": "1.0",
        "perspective": "synthetic_test",
        "population": 1,
        "horizon": 3,
        "discount_rate": 0,
        "units": "net_benefit_synthetic_utility",
        "fit_status": "valid",  # Assumption for this forward-model fixture only.
        "strategies": [
            {
                "id": strategy,
                "draws": [
                    {"draw_id": row["draw_id"], "net_benefit": row[strategy]}
                    for row in restored.iter_rows(named=True)
                ],
            }
            for strategy in ("diffusion", "alternative")
        ],
    }
    result = voiage.innovate_binding.evaluate_innovate_decision_binding(binding)
    decay = math.exp(-(rates[0] + 0.3) * 3.0)
    analytic_low = 100.0 * (1.0 - decay) / (1.0 + 0.3 / rates[0] * decay)
    reference_evpi = (16.0 - analytic_low) / 2.0
    if not math.isclose(result["evpi"], reference_evpi, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Voiage EVPI disagrees with the independent analytic oracle")
    mispaired = deepcopy(binding)
    mispaired["strategies"][1]["draws"].reverse()
    try:
        voiage.innovate_binding.evaluate_innovate_decision_binding(mispaired)
    except ValueError as exc:
        if "align" not in str(exc):
            raise
    else:
        raise ValueError("A mispaired decision binding was accepted")
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "innovate_wheel_sha256": expected_innovate,
                "voiage_wheel_sha256": expected_voiage,
                "arrow_ipc_sha256": hashlib.sha256(ipc).hexdigest(),
                "draw_ids": restored["draw_id"].to_list(),
                "strategies": result["strategy_ids"],
                "reference_evpi": reference_evpi,
                "observed_evpi": result["evpi"],
                "mispaired_draws_rejected": True,
                "utility_mapping": "one synthetic utility per adoption; alternative 16; no costs",
                "scope": "synthetic forward-model candidate; not a validated fit or release",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
