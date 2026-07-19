#!/usr/bin/env python3
"""Enforce a score from Mutmut 3.6 ``export-cicd-stats`` output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiage.mutation_policy import mutation_score_from_mapping, validate_threshold


def main() -> int:
    """Validate the official Mutmut CI/CD statistics and emit counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats", type=Path, default=Path("mutants/mutmut-cicd-stats.json")
    )
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    threshold = validate_threshold(args.threshold)
    raw = json.loads(args.stats.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("mutation statistics must be a JSON object")
    report = mutation_score_from_mapping(raw).report(threshold)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
