#!/usr/bin/env python3
"""Require one wheel with the exact ABI and platform-tag collection."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_wheel_tags(
    directory: Path, expected_abi: str, expected_platform: str
) -> Path:
    """Validate filename tags without depending on compressed-tag ordering."""
    wheels = sorted(path for path in directory.glob("*.whl") if path.is_file())
    if len(wheels) != 1:
        raise ValueError(
            f"expected exactly one wheel in {directory}, found {len(wheels)}"
        )
    wheel = wheels[0]
    fields = wheel.stem.rsplit("-", 3)
    if len(fields) != 4 or "-" not in fields[0]:
        raise ValueError(f"invalid wheel filename: {wheel.name}")
    actual_abi = f"{fields[1]}-{fields[2]}"
    if actual_abi != expected_abi:
        raise ValueError(
            f"{wheel.name}: ABI {actual_abi!r} does not match {expected_abi!r}"
        )
    actual_tags = fields[3].split(".")
    expected_tags = expected_platform.split(".")
    if (
        not all(actual_tags)
        or not all(expected_tags)
        or sorted(actual_tags) != sorted(expected_tags)
    ):
        raise ValueError(
            f"{wheel.name}: platform tags {actual_tags!r} do not match {expected_tags!r}"
        )
    return wheel


def main() -> int:
    """Validate a wheel directory using explicitly supplied expected tags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    try:
        wheel = validate_wheel_tags(args.directory, args.abi, args.platform)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Validated wheel tags: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
