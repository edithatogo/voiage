"""Regenerate the deterministic event-localized information fixture."""

# pyright: reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from voiage.methods.event_localized_information import (
    event_localized_information_value,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/event-localized-information/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def main() -> int:
    """Evaluate the normative input and write canonical sorted JSON."""
    payload = cast("dict[str, object]", json.loads(INPUT.read_text(encoding="utf-8")))
    result = event_localized_information_value(payload).to_contract_dict()
    EXPECTED.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
