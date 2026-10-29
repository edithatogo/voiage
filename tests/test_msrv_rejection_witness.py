"""Check the toolchain-selection guard used by the MSRV witness."""

from __future__ import annotations

import subprocess

import pytest

from scripts.verify_msrv_rejection import tool_version


@pytest.mark.parametrize(
    ("name", "output"),
    [("cargo", "cargo 1.86.0 (newer)"), ("rustc", "rustc 1.99.1 (Homebrew)")],
)
def test_rejects_mixed_or_newer_toolchain(
    monkeypatch: pytest.MonkeyPatch, name: str, output: str
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="must be 1.85.0"):
        tool_version("selected-tool", name)
