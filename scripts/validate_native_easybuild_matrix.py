"""Validate the complete two-generation native EasyBuild evidence matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.native_easybuild_qualification import validate_matrix


def main() -> int:
    """Validate both terminal generation receipts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--tooling-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    validate_matrix(args.matrix, args.root, args.tooling_root)
    print("Native EasyBuild matrix: PASS (2023a and 2024a)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
