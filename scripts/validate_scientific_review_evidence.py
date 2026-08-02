#!/usr/bin/env python3
"""Validate a governed scientific-review evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    validate_scientific_review_bundle,
)


def main() -> int:
    """Validate the requested JSON bundle and return a shell status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.bundle.read_text(encoding="utf-8"))
        validate_scientific_review_bundle(payload)
    except (OSError, json.JSONDecodeError, ScientificReviewEvidenceError) as error:
        print(f"scientific-review evidence invalid: {error}", file=sys.stderr)
        return 1
    print(f"scientific-review evidence valid: {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
