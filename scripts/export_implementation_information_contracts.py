#!/usr/bin/env python3
"""Regenerate the deterministic implementation-information result fixture."""

# pyright: reportAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import json
from pathlib import Path

from voiage.methods.implementation_information import implementation_information_value

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/implementation-information/v1"


def main() -> None:
    """Write the normative exact result using canonical sorted JSON."""
    request = json.loads(
        (CONTRACT / "fixtures/normative/input.json").read_text(encoding="utf-8")
    )
    result = implementation_information_value(request).to_contract_dict()
    (CONTRACT / "fixtures/normative/expected.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
