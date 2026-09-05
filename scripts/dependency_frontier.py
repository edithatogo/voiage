#!/usr/bin/env python3
"""Audit declared dependencies without confusing support floors with the lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
import urllib.request

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def declared_requirements(config: dict[str, object]) -> list[tuple[str, Requirement]]:
    """Collect core, optional, development, and build dependency declarations."""
    project = config["project"]
    assert isinstance(project, dict)
    grouped: list[tuple[str, object]] = [("core", project.get("dependencies", []))]
    optional = project.get("optional-dependencies", {})
    assert isinstance(optional, dict)
    grouped.extend((f"optional:{name}", values) for name, values in optional.items())
    dependency_groups = config.get("dependency-groups", {})
    assert isinstance(dependency_groups, dict)
    grouped.extend(
        (f"group:{name}", values) for name, values in dependency_groups.items()
    )
    build_system = config.get("build-system", {})
    assert isinstance(build_system, dict)
    grouped.append(("build", build_system.get("requires", [])))
    requirements: list[tuple[str, Requirement]] = []
    for scope, values in grouped:
        assert isinstance(values, list)
        requirements.extend(
            (scope, Requirement(item)) for item in values if isinstance(item, str)
        )
    return requirements


def release_metadata(name: str) -> dict[str, object]:
    """Return release metadata from the official PyPI JSON API."""
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = json.load(response)
    assert isinstance(payload, dict)
    return payload


def minimum_declared(requirement: Requirement) -> str | None:
    """Return the strongest declared inclusive lower bound."""
    candidates = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in {">=", "=="} and "*" not in spec.version
    ]
    return str(max(candidates)) if candidates else None


def locked_versions(lock: dict[str, object]) -> dict[str, list[str]]:
    """Index every exact version in the uv lock by normalized package name."""
    result: dict[str, set[str]] = {}
    packages = lock.get("package", [])
    assert isinstance(packages, list)
    for package in packages:
        assert isinstance(package, dict)
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            result.setdefault(canonicalize_name(name), set()).add(version)
    return {name: sorted(versions, key=Version) for name, versions in result.items()}


def usable_versions(releases: object) -> list[Version]:
    """Return sorted non-yanked final releases with valid PEP 440 versions."""
    if not isinstance(releases, dict):
        return []
    versions: list[Version] = []
    for raw_version, files in releases.items():
        try:
            version = Version(str(raw_version))
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if (
            isinstance(files, list)
            and files
            and all(
                isinstance(file, dict) and file.get("yanked", False) for file in files
            )
        ):
            continue
        versions.append(version)
    return sorted(set(versions))


def newest(values: list[Version]) -> str | None:
    """Serialize the newest version in an iterable."""
    versions = list(values)
    return str(max(versions)) if versions else None


def audit_requirement(
    scope: str,
    requirement: Requirement,
    lock_index: dict[str, list[str]],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Report support floor, lock, and release frontier as distinct facts."""
    versions = usable_versions(metadata.get("releases"))
    compatible = [version for version in versions if version in requirement.specifier]
    latest_overall = newest(versions)
    latest_compatible = newest(compatible)
    locked = lock_index.get(canonicalize_name(requirement.name), [])
    locked_satisfy = bool(locked) and all(
        Version(version) in requirement.specifier for version in locked
    )
    violations: list[str] = []
    if not locked:
        violations.append("missing_from_lock")
    elif not locked_satisfy:
        violations.append("locked_version_outside_declared_range")
    if latest_compatible is None:
        violations.append("no_compatible_final_release")

    if violations:
        support_window = "policy_violation"
    elif latest_compatible in locked:
        support_window = "locked_at_latest_compatible"
    else:
        support_window = "locked_within_supported_range"

    return {
        "scope": scope,
        "package": requirement.name,
        "specifier": str(requirement.specifier),
        "minimum_supported": minimum_declared(requirement),
        "locked": locked,
        "latest_compatible": latest_compatible,
        "latest_overall": latest_overall,
        "upper_bound_blocker": bool(
            latest_overall
            and latest_compatible
            and latest_overall != latest_compatible
            and Version(latest_overall) not in requirement.specifier
        ),
        "prereleases_allowed": requirement.specifier.prereleases is True,
        "support_window_status": support_window,
        "policy_violations": violations,
    }


def main() -> int:
    """Run the live dependency audit and write local context artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((repo / "uv.lock").read_text(encoding="utf-8"))
    requirements = declared_requirements(config)
    lock_index = locked_versions(lock)
    rows = [
        audit_requirement(
            scope, requirement, lock_index, release_metadata(requirement.name)
        )
        for scope, requirement in requirements
    ]
    violations = [
        {
            "scope": row["scope"],
            "package": row["package"],
            "reasons": row["policy_violations"],
        }
        for row in rows
        if row["policy_violations"]
    ]
    report = {
        "schema_version": "2.0",
        "requires_python": config["project"].get("requires-python"),
        "strict_policy_passed": not violations,
        "policy_violations": violations,
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
        f"Strict dependency policy: **{'pass' if report['strict_policy_passed'] else 'fail'}**",
        "",
        "| Scope | Package | Minimum supported | Locked | Latest compatible | Latest overall | Upper-bound preview | Policy |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    lines.extend(
        "| `{scope}` | `{package}` | `{minimum_supported}` | `{locked}` | "
        "`{latest_compatible}` | `{latest_overall}` | {upper} | {policy} |".format(
            **row,
            upper="yes" if row["upper_bound_blocker"] else "no",
            policy="pass" if not row["policy_violations"] else "fail",
        )
        for row in rows
    )
    (output / "dependency_frontier.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output / "dependency_frontier.md")
    return 2 if args.strict and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
