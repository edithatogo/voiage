"""Export heterogeneity-value v1 schemas and the deterministic result."""

# pyright: reportAny=false, reportExplicitAny=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from voiage.contracts.heterogeneity_value import (
    HETEROGENEITY_VALUE_INPUT_SCHEMA_V1,
    HETEROGENEITY_VALUE_RESULT_SCHEMA_V1,
)
from voiage.methods.heterogeneity_value import heterogeneity_value_decomposition

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/heterogeneity-value/v1"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Write installed schemas and the normative exact-enumeration output."""
    _write(CONTRACT / "schemas/input.schema.json", HETEROGENEITY_VALUE_INPUT_SCHEMA_V1)
    _write(
        CONTRACT / "schemas/result.schema.json", HETEROGENEITY_VALUE_RESULT_SCHEMA_V1
    )
    input_path = CONTRACT / "fixtures/normative/input.json"
    payload = cast(
        "dict[str, object]", json.loads(input_path.read_text(encoding="utf-8"))
    )
    result: dict[str, Any] = heterogeneity_value_decomposition(
        payload
    ).to_contract_dict()
    _write(CONTRACT / "fixtures/normative/expected.json", result)


if __name__ == "__main__":
    main()
