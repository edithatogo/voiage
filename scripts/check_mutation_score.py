#!/usr/bin/env python3
"""Enforce a score from Mutmut 3.6 ``export-cicd-stats`` output."""

# pyright: reportAny=false, reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from voiage.mutation_policy import mutation_score_from_mapping, validate_threshold


def main() -> int:
    """Validate the official Mutmut CI/CD statistics and emit counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats", type=Path, default=Path("mutants/mutmut-cicd-stats.json")
    )
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-stats", type=Path)
    args = parser.parse_args()
    threshold = validate_threshold(args.threshold)
    score = mutation_score_from_mapping(_read_stats(args.stats))
    baseline = (
        mutation_score_from_mapping(_read_stats(args.baseline_stats))
        if args.baseline_stats is not None
        else None
    )
    report = score.report(threshold, baseline=baseline)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0 if report["passed"] else 2


def _read_stats(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("mutation statistics must be a JSON object")
    return cast("dict[str, object]", raw)


if __name__ == "__main__":
    raise SystemExit(main())
