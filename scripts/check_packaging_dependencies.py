"""Fail closed when runtime packaging surfaces disagree with the reviewed set."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import tomllib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
import yaml


def requirements(values: list[str]) -> dict[str, str]:
    """Normalize names and specifiers without confusing tools with runtime roots."""
    result = {}
    for value in values:
        try:
            requirement = Requirement(value.strip())
        except InvalidRequirement:
            if not value.strip() or value.strip().startswith("."):
                continue  # Local project requirements are validated separately.
            raise
        extras = (
            f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        )
        marker = f"; {requirement.marker}" if requirement.marker else ""
        result[canonicalize_name(requirement.name)] = (
            extras + str(requirement.specifier) + marker
        )
    return result


def validate(repo: Path) -> dict[str, object]:
    """Compare published metadata, Conda, tox lanes and dependency observations."""
    policy = json.loads((repo / "specs/dependency-runtime-policy.json").read_text())
    config = tomllib.loads((repo / "pyproject.toml").read_text())
    expected = requirements(policy["runtime"])
    errors: list[str] = []

    def compare(label: str, actual: dict[str, str], wanted: dict[str, str]) -> None:
        if actual != wanted:
            errors.append(f"{label}: expected {wanted!r}; found {actual!r}")

    compare("PyPI runtime", requirements(config["project"]["dependencies"]), expected)
    if config["project"]["requires-python"] != policy["requires_python"]:
        errors.append("PyPI Python range differs from the qualified interpreter policy")
    conda = (repo / "conda-recipe/meta.yaml").read_text()
    run = conda.split("  run:\n", 1)[1].split("\ntest:", 1)[0]
    conda_requirements = requirements(
        [
            line.strip()[2:].replace(" ", "")
            for line in run.splitlines()
            if line.strip().startswith("- ")
        ]
    )
    python = conda_requirements.pop("python", None)
    compare("Conda runtime", conda_requirements, expected)
    if python != str(Requirement("python" + policy["requires_python"]).specifier):
        errors.append(
            "Conda Python range differs from the qualified interpreter policy"
        )
    if "skip: true  # [py<315]" not in conda:
        errors.append("Conda selector must exclude older Python runners")
    tox = configparser.ConfigParser(interpolation=None)
    tox.read(repo / "tox.ini")
    for section in tox.sections():
        deps = requirements(tox.get(section, "deps", fallback="").splitlines())
        if {"pandas", "xarray"} & deps.keys():
            errors.append(
                f"{section}: legacy runtime declaration outside producer group"
            )
    for name in ("polars", "pyarrow"):
        minimum = requirements(tox["testenv:min_versions"]["deps"].splitlines())
        maximum = requirements(tox["testenv:max_versions"]["deps"].splitlines())
        compare(
            f"tox {name} floor",
            {name: minimum.get(name, "")},
            {name: "==" + policy["floors"][name]},
        )
        compare(
            f"tox {name} latest", {name: maximum.get(name, "")}, {name: expected[name]}
        )
    if "." not in tox["testenv:typecheck"]["deps"].split():
        errors.append("Typecheck must install the canonical project runtime")
    preview = yaml.safe_load(
        (repo / ".github/workflows/dependency-preview.yml").read_text()
    )
    candidates = preview["jobs"]["major-dependency-observation"]["strategy"]["matrix"][
        "include"
    ]
    observed = requirements([row["requirement"] for row in candidates])
    if {"pandas", "xarray"} & observed.keys():
        errors.append("Dependency preview observes retired runtime packages")
    for name in ("polars", "pyarrow"):
        compare(
            f"preview {name}", {name: observed.get(name, "")}, {name: expected[name]}
        )
    errors.extend(
        "Published extras must not declare legacy tabular engines"
        for values in config["project"]["optional-dependencies"].values()
        if {"pandas", "xarray"} & requirements(values).keys()
    )
    if config.get("dependency-groups", {}).get("producer-tests") != [
        "pandas>=3.1.0,<4"
    ]:
        errors.append("Producer tests require the explicit unpublished qualified group")
    if config.get("tool", {}).get("uv", {}).get("default-groups") != []:
        errors.append("Default installs must not select producer tooling")
    return {
        "schema_version": "1.0",
        "passed": not errors,
        "errors": errors,
        "runtime": expected,
        "producer_tooling": "unpublished, explicit opt-in; excluded from wheel requirements",
    }


def main() -> int:
    """Write a reviewable runtime dependency agreement result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    report = validate(args.repo)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
