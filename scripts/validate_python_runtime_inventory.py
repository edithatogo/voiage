#!/usr/bin/env python3
"""Validate the machine-readable Python runtime inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def validate(repo_root: Path) -> list[str]:
    """Return policy violations in the machine-readable runtime inventory."""
    manifest_path = repo_root / "specs/v2/python-runtime-inventory.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    categories = manifest["categories"]
    retained = set(manifest["v2_retained_categories"])
    transitional = manifest["transitional_kernel_modules"]
    errors: list[str] = []
    if "transitional_numerical_core" in retained:
        errors.append("transitional_numerical_core cannot be in v2_retained_categories")
    if transitional.get("authoritative_owner") != "rust":
        errors.append(
            "transitional numerical kernels must declare Rust as authoritative"
        )
    if transitional.get("python_role") != "rust_facade_and_orchestration_only":
        errors.append(
            "retired numerical modules must declare Python as facade and orchestration only"
        )
    roots = [
        (root, category)
        for category, data in categories.items()
        for root in data["roots"]
    ]
    errors.extend(
        f"transitional kernel module is missing: {module}"
        for module in transitional["modules"]
        if not (repo_root / module).is_file()
    )
    for path in sorted((repo_root / "voiage").rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        matches = [
            category
            for root, category in roots
            if relative == root or relative.startswith(root)
        ]
        if len(matches) != 1:
            errors.append(
                f"{relative}: expected exactly one inventory category, found {matches}"
            )
    if any(category not in categories for category in retained):
        errors.append("v2_retained_categories contains an unknown category")
    return errors


def main() -> int:
    """Validate the inventory at the requested repository root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = validate(args.repo.resolve())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Python runtime inventory is complete and unambiguous.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
