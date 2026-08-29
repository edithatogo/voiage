#!/usr/bin/env python3
"""Execute the rights-cleared synthetic research-workflow corpus."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from voiage.methods.basic import evpi
from voiage.schema import ValueArray

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "specs/workflows/research-workflow-corpus-v1.json"


def validate(corpus_path: Path = DEFAULT_CORPUS) -> list[str]:
    """Run every workflow and return deterministic validation errors."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for workflow in corpus["workflows"]:
        source = json.loads((ROOT / workflow["input"]).read_text(encoding="utf-8"))
        expected = json.loads(
            (ROOT / workflow["expected_result"]).read_text(encoding="utf-8")
        )
        if source.get("source") != {
            "kind": "synthetic",
            "rights": "repository-authored",
        }:
            errors.append(
                f"{workflow['id']}: source is not rights-cleared synthetic data"
            )
            continue
        values = ValueArray.from_numpy(
            np.asarray(source["net_benefit"], dtype=float),
            strategy_names=source["strategy_names"],
        )
        value = float(evpi(values))
        means = values.numpy_values.mean(axis=0)
        prior = source["strategy_names"][int(means.argmax())]
        report = f"{workflow['id']}: EVPI {value:.6f} {source['unit']}"
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "workflow_id": workflow["id"],
            "evpi": value,
            "prior_strategy": prior,
            "report": report,
            "stages": workflow["stages"],
        }
        serialized = json.loads(json.dumps(result, allow_nan=False, sort_keys=True))
        if not math.isclose(
            serialized["evpi"], expected["evpi"], rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"{workflow['id']}: EVPI differs from expected result")
        if serialized["prior_strategy"] != expected["prior_strategy"]:
            errors.append(f"{workflow['id']}: prior strategy differs")
        if expected["report_contains"] not in serialized["report"]:
            errors.append(f"{workflow['id']}: report assertion differs")
        if serialized["stages"] != [
            "source",
            "ingestion",
            "analysis",
            "serialization",
            "report",
        ]:
            errors.append(f"{workflow['id']}: workflow stages are incomplete")
    return errors


def main() -> int:
    """Validate the configured corpus and emit a compact JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    errors = validate(args.corpus)
    print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
