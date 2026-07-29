"""Drift checks for the registry-generated C ABI capability document."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]


def test_generated_ffi_capabilities_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_ffi_capabilities.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
