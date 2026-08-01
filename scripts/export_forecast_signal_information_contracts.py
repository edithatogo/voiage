"""Export portable forecast-signal schemas and the normative result fixture."""

# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import json
from pathlib import Path

from voiage.contracts.forecast_signal_information import (
    FORECAST_SIGNAL_INFORMATION_INPUT_SCHEMA_V1,
    FORECAST_SIGNAL_INFORMATION_RESULT_SCHEMA_V1,
)
from voiage.methods.forecast_signal_information import (
    forecast_signal_information_value,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/forecast-signal-information/v1"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Write schemas and the exact normative evaluator output."""
    _write(
        CONTRACT / "schemas/forecast-signal-information-input.schema.json",
        FORECAST_SIGNAL_INFORMATION_INPUT_SCHEMA_V1,
    )
    _write(
        CONTRACT / "schemas/forecast-signal-information-result.schema.json",
        FORECAST_SIGNAL_INFORMATION_RESULT_SCHEMA_V1,
    )
    request = json.loads(
        (CONTRACT / "fixtures/normative/input.json").read_text(encoding="utf-8")
    )
    result = forecast_signal_information_value(request).to_contract_dict()
    _write(CONTRACT / "fixtures/normative/expected.json", result)


if __name__ == "__main__":
    main()
