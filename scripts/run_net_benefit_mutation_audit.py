#!/usr/bin/env python3
"""Run a bounded, dependency-free mutation audit of the Rust net-benefit kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

MUTANTS = {
    "subtract-cost-to-add-cost": (
        "effects[value_index] * threshold - costs[value_index]",
        "effects[value_index] * threshold + costs[value_index]",
    ),
    "finite-result-guard-removed": (
        "if value.is_finite() {",
        "if true {",
    ),
    "scalar-threshold-zeroed": (
        "push_value(value_index, willingness_to_pay[0])?;",
        "push_value(value_index, 0.0)?;",
    ),
    "sample-threshold-ownership-removed": (
        "willingness_to_pay[sample * threshold_count + threshold]",
        "willingness_to_pay[threshold]",
    ),
}


def _run(repo: Path, output: Path) -> int:
    source = repo / "rust/crates/voiage-numerics/src/net_benefit.rs"
    original = source.read_text(encoding="utf-8")
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="voiage-net-benefit-mutants-") as temporary:
        temporary_root = Path(temporary)
        workspace = temporary_root / "rust"
        shutil.copytree(
            repo / "rust", workspace, ignore=shutil.ignore_patterns("target")
        )
        mutated_source = workspace / "crates/voiage-numerics/src/net_benefit.rs"
        for mutant_id, (before, after) in MUTANTS.items():
            if original.count(before) != 1:
                raise RuntimeError(
                    f"{mutant_id}: expected exactly one source mutation site"
                )
            mutated_source.write_text(
                original.replace(before, after), encoding="utf-8", newline="\n"
            )
            command = [
                "cargo",
                "test",
                "-p",
                "voiage-numerics",
                "--lib",
                "--test",
                "net_benefit",
            ]
            completed = subprocess.run(  # noqa: S603 - fixed local cargo invocation
                command,
                cwd=workspace,
                check=False,
                capture_output=True,
                text=True,
            )
            results.append(
                {
                    "mutant_id": mutant_id,
                    "status": "killed" if completed.returncode else "survived",
                    "test_command": " ".join(command),
                }
            )
    killed = sum(result["status"] == "killed" for result in results)
    report = {
        "schema_version": 1,
        "scope": "rust/crates/voiage-numerics/src/net_benefit.rs",
        "mutants": results,
        "killed": killed,
        "total": len(results),
        "score_percent": 100.0 * killed / len(results),
        "passed": killed == len(results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def main() -> int:
    """Run the bounded net-benefit mutation audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmarks/mutation-net-benefit.json"),
    )
    arguments = parser.parse_args()
    return _run(arguments.repo.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
