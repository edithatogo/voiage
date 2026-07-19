#!/usr/bin/env python3
"""Bind mutation score, debt, and density to an immutable source/config cohort."""

# pyright: reportAny=false, reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import cast

from voiage.mutation_policy import mutation_score_from_mapping, validate_threshold


def _object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", value)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def cohort_identity(repo: Path, config_path: Path) -> dict[str, object]:
    """Return canonical Mutmut configuration and source identities."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    mutmut = cast(
        "dict[str, object]", cast("dict[str, object]", config["tool"])["mutmut"]
    )
    targets = cast("list[str]", mutmut["only_mutate"])
    sources: list[dict[str, object]] = []
    logical_lines = 0
    for relative in sorted(targets):
        path = repo / relative
        data = path.read_bytes()
        logical_lines += sum(
            1
            for line in data.decode("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        sources.append({"path": relative, "sha256": sha256(data).hexdigest()})
    configuration_sha256 = sha256(_canonical(mutmut)).hexdigest()
    cohort = {
        "tool": "mutmut",
        "configuration_sha256": configuration_sha256,
        "sources": sources,
    }
    return {
        **cohort,
        "source_logical_lines": logical_lines,
        "cohort_sha256": sha256(_canonical(cohort)).hexdigest(),
    }


def evaluate_cohort(
    stats: dict[str, object],
    baseline: dict[str, object],
    identity: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    """Evaluate score and two debt ratchets only for an identical cohort."""
    if baseline.get("schema_version") != "1.0.0":
        raise ValueError("unsupported mutation cohort baseline schema")
    provenance = cast("dict[str, object]", baseline.get("promotion_provenance"))
    provenance_valid = (
        provenance.get("human_approved") is True
        and isinstance(provenance.get("run_id"), int)
        and isinstance(provenance.get("commit"), str)
        and len(cast("str", provenance.get("commit"))) == 40
        and isinstance(provenance.get("evidence_url"), str)
    )
    expected_identity = cast("dict[str, object]", baseline["cohort"])
    identity_matches = identity == expected_identity
    score = mutation_score_from_mapping(stats)
    baseline_score = mutation_score_from_mapping(
        cast("dict[str, object]", baseline["stats"])
    )
    score_report = score.report(validate_threshold(threshold), baseline=baseline_score)
    debt = score.eligible - score.killed
    debt_density = debt / score.eligible if score.eligible else 1.0
    logical_lines = int(cast("int", identity["source_logical_lines"]))
    mutants_per_kloc = score.eligible * 1000.0 / logical_lines if logical_lines else 0.0
    policy = cast("dict[str, object]", baseline["policy"])
    maximum_debt = int(cast("int", policy["maximum_absolute_debt"]))
    maximum_density = float(cast("int | float", policy["maximum_debt_density"]))
    minimum_score = float(cast("int | float", policy["minimum_score_percent"]))
    passed = (
        identity_matches
        and provenance_valid
        and threshold >= minimum_score
        and bool(score_report["passed"])
        and debt <= maximum_debt
        and debt_density <= maximum_density
    )
    return {
        "schema_version": "1.0.0",
        "cohort": identity,
        "expected_cohort_sha256": expected_identity.get("cohort_sha256"),
        "identity_matches": identity_matches,
        "promotion_provenance": provenance,
        "promotion_provenance_valid": provenance_valid,
        "score": score_report,
        "promoted_minimum_score_percent": minimum_score,
        "debt": {
            "absolute": debt,
            "maximum_absolute": maximum_debt,
            "density": round(debt_density, 6),
            "maximum_density": maximum_density,
            "mutants_per_kloc": round(mutants_per_kloc, 3),
        },
        "passed": passed,
    }


def main() -> int:
    """Evaluate current Mutmut output against its promoted cohort baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats", type=Path, default=Path("mutants/mutmut-cicd-stats.json")
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = evaluate_cohort(
        _object(args.stats),
        _object(args.baseline),
        cohort_identity(repo, repo / args.config),
        args.threshold,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
