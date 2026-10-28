"""Regression tests for the installed-wheel test runner."""

from importlib.machinery import EXTENSION_SUFFIXES, ModuleSpec
from pathlib import Path
import sys

import pytest

from scripts import run_installed_pytest as runner


def test_missing_wheel_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match="installed voiage wheel is unavailable"):
        runner.main()


def test_checkout_package_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    package = Path(runner.__file__).resolve().parents[1] / "voiage"
    spec = ModuleSpec("voiage", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: spec)
    with pytest.raises(SystemExit, match="checkout shadows"):
        runner.main()


def test_installed_wheel_drives_coverage_and_import_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "site-packages" / "voiage"
    spec = ModuleSpec("voiage", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: spec)
    package.mkdir(parents=True)
    (package / f"_core{EXTENSION_SUFFIXES[0]}").touch()
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["runner", "tests/", "--cov-installed-voiage"])
    calls: list[list[str]] = []
    monkeypatch.setattr(pytest, "main", lambda args: calls.append(args) or 0)

    assert runner.main() == 0
    assert calls == [["--import-mode=importlib", "tests/", f"--cov={package}"]]
    assert sys.path[-1] == str(Path(runner.__file__).resolve().parents[1])


def test_other_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = ModuleSpec("voiage", loader=None, is_package=True)
    spec.submodule_search_locations = [str(tmp_path / "other" / "voiage")]
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: spec)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "active"))
    with pytest.raises(SystemExit, match="outside the active environment"):
        runner.main()


def test_missing_native_extension_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = ModuleSpec("voiage", loader=None, is_package=True)
    spec.submodule_search_locations = [str(tmp_path / "voiage")]
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: spec)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    with pytest.raises(SystemExit, match="no native extension"):
        runner.main()
