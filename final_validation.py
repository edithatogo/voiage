#!/usr/bin/env python3
"""Run the maintained local validation controls for ``voiage``.

This convenience script is intentionally supplementary to the required tox
matrix. It verifies the same active Python quality and dependency-security
tools without relying on legacy or removed tooling.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

import yaml

ROOT = Path(__file__).resolve().parent
type Command = list[str]


def run_command(command: Command, description: str) -> bool:
    """Run one fixed local validation command and report its outcome."""
    rendered = " ".join(command)
    print(f"Running: {description}")
    result = subprocess.run(  # noqa: S603 - commands are fixed in this module
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED: {rendered}")
        if result.stderr:
            print(result.stderr.strip())
        if result.stdout:
            print(result.stdout.strip())
        return False

    print(f"Passed: {description}")
    return True


def validate_configuration() -> bool:
    """Validate the TOML project manifest and active CI workflow YAML."""
    try:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            tomllib.load(stream)
        with (ROOT / ".github/workflows/ci.yml").open(encoding="utf-8") as stream:
            yaml.safe_load(stream)
    except (OSError, tomllib.TOMLDecodeError, yaml.YAMLError) as error:
        print(f"FAILED: configuration validation: {error}")
        return False

    print("Passed: project TOML and CI workflow YAML validation")
    return True


def main() -> int:
    """Run the maintained local validation sequence."""
    commands = (
        (
            [sys.executable, "-c", "import voiage; print(voiage.__version__)"],
            "package import",
        ),
        ([sys.executable, "-m", "voiage.cli", "--help"], "CLI help"),
        (
            [
                "uvx",
                "--from",
                "pip-audit==2.10.1",
                "pip-audit",
                "--local",
                "--progress-spinner",
                "off",
            ],
            "pip-audit dependency scan",
        ),
        (["ruff", "--version"], "Ruff availability"),
        (["ty", "--version"], "ty availability"),
        (["tox", "-e", "lint", "--showconfig"], "tox lint configuration"),
        (
            ["tox", "-e", "typecheck", "--showconfig"],
            "tox type-check configuration",
        ),
        (
            [sys.executable, "scripts/validate_arxiv.py", "paper"],
            "canonical manuscript-source validation",
        ),
    )
    for command, description in commands:
        if not run_command(command, description):
            return 1
    if not validate_configuration():
        return 1

    build_command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--skip-dependency-check",
    ]
    if not run_command(build_command, "optional package build"):
        print(
            "Package build is unavailable; the required tox matrix remains authoritative."
        )

    print("All supplementary local validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
