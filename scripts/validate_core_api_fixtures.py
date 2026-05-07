#!/usr/bin/env python3
"""Smoke-validate the core API fixture manifest and artifact layout."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_core_api_contract as validator  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(validator.REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    """Validate the committed core API fixture catalog layout."""
    validator.validate_fixture_catalog_layout()
    print(f"validated {_display_path(validator.FIXTURE_MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
