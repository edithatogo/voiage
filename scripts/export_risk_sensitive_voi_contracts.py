"""Export the installed issue #570 schemas into the portable frontier tree."""

from __future__ import annotations

import json
from pathlib import Path

from voiage.contracts.risk_sensitive_voi import (
    RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1,
    RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1,
)

ROOT = Path(__file__).parents[1]
SCHEMAS = ROOT / "specs/frontier/risk-sensitive-constrained-voi/v1/schemas"


def main() -> int:
    """Write deterministic portable schema files."""
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("risk-sensitive-voi-input.schema.json", RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1),
        ("risk-sensitive-voi-result.schema.json", RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1),
    ):
        (SCHEMAS / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
