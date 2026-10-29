#!/usr/bin/env python3
"""Run checkout tests against the wheel installed in the active environment.

Invoke with ``python -I`` so the checkout cannot shadow the installed package.
The checkout is appended for test helpers, and pytest importlib mode keeps
parallel workers from prepending it again.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def main() -> int:
    """Resolve the installed package before exposing repository test helpers."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("voiage")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("The installed voiage wheel is unavailable")
    package = Path(next(iter(spec.submodule_search_locations))).resolve()
    if package == (root / "voiage").resolve():
        raise SystemExit(
            "The checkout shadows the installed voiage wheel; use python -I"
        )
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise SystemExit("The voiage package is outside the active environment")
    if not any(
        (package / f"_core{suffix}").is_file()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise SystemExit("The installed voiage wheel has no native extension")
    sys.path.append(str(root))

    import pytest

    args = [
        f"--cov={package}" if arg == "--cov-installed-voiage" else arg
        for arg in sys.argv[1:]
    ]
    return pytest.main(["--import-mode=importlib", *args])


if __name__ == "__main__":
    raise SystemExit(main())
