"""Regenerate the outcome-conditional sample-information contract bundle."""

# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from voiage.contracts.outcome_conditional_sample_information import (
    OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1,
    OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1,
)
from voiage.methods.outcome_conditional_sample_information import (
    outcome_conditional_sample_information_value,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/outcome-conditional-sample-information/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Write strict schemas and the deterministic normative result."""
    _write(
        CONTRACT / "schemas/input.schema.json",
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1,
    )
    _write(
        CONTRACT / "schemas/result.schema.json",
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1,
    )
    payload = cast("dict[str, object]", json.loads(INPUT.read_text(encoding="utf-8")))
    result = outcome_conditional_sample_information_value(payload).to_contract_dict()
    _write(EXPECTED, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
