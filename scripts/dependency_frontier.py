#!/usr/bin/env python3
"""Audit declared direct dependencies against the live PyPI release frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any, cast
import urllib.request

from packaging.requirements import Requirement
from packaging.version import Version


def declared_requirements(config: dict[str, Any]) -> list[tuple[str, Requirement]]:
    """Collect core, optional, development, and build dependency declarations."""
    project = config["project"]
    assert isinstance(project, dict)
    grouped: list[tuple[str, list[str]]] = [
        ("core", cast("list[str]", project.get("dependencies", [])))
    ]
    grouped.extend(
        (f"optional:{name}", cast("list[str]", values))
        for name, values in project.get("optional-dependencies", {}).items()
    )
    grouped.extend(
        (f"group:{name}", cast("list[str]", values))
        for name, values in config.get("dependency-groups", {}).items()
    )
    grouped.append(
        (
            "build",
            cast("list[str]", config.get("build-system", {}).get("requires", [])),
        )
    )
    return [
        (scope, Requirement(item))
        for scope, values in grouped
        for item in values
        if isinstance(item, str)
    ]


def pypi_releases(name: str) -> tuple[str, list[str]]:
    """Return the current version and release set from the official JSON API."""
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
        payload = json.load(response)
    return str(payload["info"]["version"]), list(payload["releases"])


def minimum_declared(requirement: Requirement) -> str | None:
    """Return the strongest declared inclusive lower bound."""
    candidates = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in {">=", "=="} and "*" not in spec.version
    ]
    return str(max(candidates)) if candidates else None


def latest_compatible(requirement: Requirement, releases: list[str]) -> str | None:
    """Return the newest stable release admitted by the declared range."""
    candidates = []
    for release in releases:
        try:
            version = Version(release)
        except ValueError:
            continue
        if (
            not version.is_prerelease
            and not version.is_devrelease
            and requirement.specifier.contains(version, prereleases=False)
        ):
            candidates.append(version)
    return str(max(candidates)) if candidates else None


def locked_versions(lock: dict[str, Any]) -> dict[str, str]:
    """Return the greatest resolved version for every registry package."""
    resolved: dict[str, Version] = {}
    for package in lock.get("package", []):
        if not isinstance(package, dict) or not isinstance(package.get("name"), str):
            continue
        version_text = package.get("version")
        if not isinstance(version_text, str):
            continue
        try:
            version = Version(version_text)
        except ValueError:
            continue
        normalized = package["name"].lower().replace("_", "-")
        resolved[normalized] = max(version, resolved.get(normalized, version))
    return {name: str(version) for name, version in resolved.items()}


def main() -> int:
    """Run the live dependency audit and write local context artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((repo / "uv.lock").read_text(encoding="utf-8"))
    locked = locked_versions(lock)
    project = config["project"]
    requirements = declared_requirements(config)
    rows = []
    for scope, requirement in requirements:
        latest, releases = pypi_releases(requirement.name)
        compatible = latest_compatible(requirement, releases)
        declared = minimum_declared(requirement)
        normalized = requirement.name.lower().replace("_", "-")
        resolved = locked.get(normalized)
        current = (
            compatible is not None
            and resolved is not None
            and Version(resolved) >= Version(compatible)
        )
        rows.append(
            {
                "scope": scope,
                "package": requirement.name,
                "declared_minimum": declared,
                "latest": latest,
                "latest_compatible": compatible,
                "locked": resolved,
                "at_frontier": current,
                "newer_release_outside_range": (
                    compatible is not None and Version(latest) > Version(compatible)
                ),
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
        "| Scope | Package | Declared minimum | Locked | Latest compatible | PyPI latest | Current |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    lines.extend(
        f"| `{row['scope']}` | `{row['package']}` | `{row['declared_minimum']}` | `{row['locked']}` | `{row['latest_compatible']}` | `{row['latest']}` | {'yes' if row['at_frontier'] else 'no'} |"
        for row in rows
    )
    (output / "dependency_frontier.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output / "dependency_frontier.md")
    return 2 if args.strict and not report["all_direct_dependencies_at_frontier"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
