#!/usr/bin/env python3
"""Record or compare normalized wheel and sdist identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from voiage.c15_reproducibility import (
    ArtifactMismatchError,
    compare_digest_reports,
    normalized_archive_report,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"digest report must be an object: {path}")
    return cast("dict[str, object]", value)


def _single(directory: Path, kind: str) -> Path:
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (
            (kind == "wheel" and path.suffix == ".whl")
            or (kind == "sdist" and path.name.endswith(".tar.gz"))
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one {kind} in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def main() -> int:
    """Record or compare one normalized artifact report."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--archive-dir", type=Path, required=True)
    record.add_argument("--kind", choices=["wheel", "sdist"], required=True)
    record.add_argument("--runner", required=True)
    record.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "record":
            payload = normalized_archive_report(
                _single(args.archive_dir, args.kind), runner=args.runner
            )
        else:
            payload = compare_digest_reports(_load(args.left), _load(args.right))
        _write(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    except (ArtifactMismatchError, OSError, TypeError, ValueError) as exc:
        failure: dict[str, object] = {
            "schema_version": "1.0.0",
            "passed": False,
            "operation": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            _write(args.output, failure)
        except OSError as write_error:
            print(f"C15 failure evidence could not be retained: {write_error}")
        print(f"C15 artifact assurance failed: {exc}")
        return 2
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
