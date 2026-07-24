#!/usr/bin/env python3
"""Fail when the reproducible VOI software/literature census is overdue."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LANDSCAPE = ROOT / "specs" / "software-landscape"


def main() -> int:
    """Validate every landscape review deadline against the requested date."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Date used for deterministic testing (YYYY-MM-DD).",
    )
    args = parser.parse_args()
    software = json.loads((LANDSCAPE / "registry.json").read_text(encoding="utf-8"))
    adjacent = json.loads(
        (LANDSCAPE / "adjacent-method-dispositions.json").read_text(encoding="utf-8")
    )
    methods = json.loads(
        (LANDSCAPE / "method-evidence.json").read_text(encoding="utf-8")
    )
    deadlines = {
        "software landscape": date.fromisoformat(software["review_due"]),
        "adjacent-method dispositions": date.fromisoformat(adjacent["review_due"]),
        "method evidence": date.fromisoformat(methods["review_due"]),
    }
    overdue = {
        name: deadline.isoformat()
        for name, deadline in deadlines.items()
        if args.as_of > deadline
    }
    if overdue:
        for name, deadline in overdue.items():
            print(f"{name} review overdue after {deadline}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
