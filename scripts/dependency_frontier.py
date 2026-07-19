#!/usr/bin/env python3
"""Audit declared direct dependencies against the live PyPI release frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
import urllib.request

from packaging.requirements import Requirement
from packaging.version import Version


def latest_version(name: str) -> str:
    """Return the current PyPI version using the official JSON API."""
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
        return str(json.load(response)["info"]["version"])


def minimum_declared(requirement: Requirement) -> str | None:
    """Return the strongest declared inclusive lower bound."""
    candidates = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in {">=", "=="} and "*" not in spec.version
    ]
    return str(max(candidates)) if candidates else None


def main() -> int:
    """Run the live dependency audit and write local context artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements = [Requirement(item) for item in project.get("dependencies", [])]
    rows = []
    for requirement in requirements:
        latest = latest_version(requirement.name)
        declared = minimum_declared(requirement)
        current = declared is not None and Version(declared) >= Version(latest)
        rows.append(
            {
                "package": requirement.name,
                "declared_minimum": declared,
                "latest": latest,
                "at_frontier": current,
                "specifier": str(requirement.specifier),
            }
        )
    report = {
        "schema_version": "1.0",
        "requires_python": project.get("requires-python"),
        "all_direct_dependencies_at_frontier": all(row["at_frontier"] for row in rows),
        "dependencies": rows,
    }
    output = repo / ".conductor" / "local"
    output.mkdir(parents=True, exist_ok=True)
    (output / "dependency_frontier.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Dependency frontier",
        "",
        f"Python: `{report['requires_python']}`",
        f"All direct dependencies current: **{report['all_direct_dependencies_at_frontier']}**",
        "",
        "| Package | Declared minimum | PyPI latest | Current |",
        "|---|---:|---:|:---:|",
    ]
    lines.extend(
        f"| `{row['package']}` | `{row['declared_minimum']}` | `{row['latest']}` | {'yes' if row['at_frontier'] else 'no'} |"
        for row in rows
    )
    (output / "dependency_frontier.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output / "dependency_frontier.md")
    return 2 if args.strict and not report["all_direct_dependencies_at_frontier"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
