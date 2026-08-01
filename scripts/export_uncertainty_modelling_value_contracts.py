"""Export the v1 schemas and deterministic finite reference results."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast

from voiage.contracts.uncertainty_modelling_value import (
    UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1,
    UNCERTAINTY_MODELLING_VALUE_RESULT_SCHEMA_V1,
)
from voiage.methods.uncertainty_modelling_value import value_of_uncertainty_modelling

type _JsonObject = dict[str, Any]

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/uncertainty-modelling-value/v1"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Write schemas and exact expected outputs from normative inputs."""
    _write(
        CONTRACT / "schemas/input.schema.json",
        UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1,
    )
    _write(
        CONTRACT / "schemas/result.schema.json",
        UNCERTAINTY_MODELLING_VALUE_RESULT_SCHEMA_V1,
    )
    for name in ("two-stage-nonlinear", "multistage"):
        input_path = CONTRACT / f"fixtures/normative/{name}-input.json"
        payload = cast(
            "_JsonObject", json.loads(input_path.read_text(encoding="utf-8"))
        )
        _write(
            CONTRACT / f"fixtures/normative/{name}-expected.json",
            value_of_uncertainty_modelling(payload).to_contract_dict(),
        )
    source = cast(
        "_JsonObject",
        json.loads(
            (CONTRACT / "fixtures/normative/two-stage-nonlinear-input.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    infeasible = deepcopy(source)
    infeasible["analysis_id"] = "infeasible-induced-recourse"
    risky = next(
        policy
        for policy in infeasible["policies"]
        if policy["policy_id"] == "risky-policy"
    )
    high = next(
        outcome for outcome in risky["state_outcomes"] if outcome["state_id"] == "high"
    )
    high.update(
        {"feasible": False, "objective_value": None, "recourse_status": "infeasible"}
    )
    _write(CONTRACT / "fixtures/normative/infeasible-recourse-input.json", infeasible)
    _write(
        CONTRACT / "fixtures/normative/infeasible-recourse-expected.json",
        value_of_uncertainty_modelling(infeasible).to_contract_dict(),
    )


if __name__ == "__main__":
    main()
