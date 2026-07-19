#!/usr/bin/env python3
"""Run a self-contained strict mutation lane over production C13 invariants."""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from voiage.mutation_policy import mutation_score_from_mapping, validate_threshold

CRITICAL_MUTMUT_CONFIG = """\
[tool.pytest.ini_options]
pythonpath = ["."]

[tool.mutmut]
source_paths = ["voiage/"]
only_mutate = ["voiage/contracts/critical_invariants.py"]
pytest_add_cli_args = ["-x", "--no-cov", "-p", "no:cacheprovider"]
pytest_add_cli_args_test_selection = ["tests/test_critical_invariants.py"]
"""


def mutation_report(raw: dict[str, object], *, threshold: float) -> dict[str, object]:
    """Return the standard fail-closed score for the strict lane."""
    return mutation_score_from_mapping(raw).report(validate_threshold(threshold))


def _run(repo: Path, output: Path, threshold: float) -> int:
    with tempfile.TemporaryDirectory(prefix="voiage-critical-mutation-") as temp:
        sandbox = Path(temp)
        package = sandbox / "voiage"
        contracts = package / "contracts"
        tests = sandbox / "tests"
        contracts.mkdir(parents=True)
        tests.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8", newline="\n")
        (contracts / "__init__.py").write_text("", encoding="utf-8", newline="\n")
        shutil.copy2(
            repo / "voiage/contracts/critical_invariants.py",
            contracts / "critical_invariants.py",
        )
        shutil.copy2(repo / "tests/test_critical_invariants.py", tests)
        (sandbox / "pyproject.toml").write_text(
            CRITICAL_MUTMUT_CONFIG, encoding="utf-8", newline="\n"
        )
        subprocess.run([sys.executable, "-m", "mutmut", "run"], cwd=sandbox, check=True)
        subprocess.run(
            [sys.executable, "-m", "mutmut", "export-cicd-stats"],
            cwd=sandbox,
            check=True,
        )
        raw = json.loads(
            (sandbox / "mutants/mutmut-cicd-stats.json").read_text(encoding="utf-8")
        )
    if not isinstance(raw, dict):
        raise TypeError("Mutmut statistics must be a JSON object")
    report = mutation_report(raw, threshold=threshold)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 2


def main() -> int:
    """Run the strict lane in an isolated Linux CI sandbox."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/mutation-critical.json"),
    )
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()
    return _run(args.repo.resolve(), args.output.resolve(), args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
