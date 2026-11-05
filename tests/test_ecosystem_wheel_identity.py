"""Test installed-adapter evidence against a shadowing package directory."""

from hashlib import sha256
import runpy
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_unreviewed_subpackage_cannot_shadow_verified_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verify = runpy.run_path(
        "conductor/tracks/scientific_ecosystem_alignment_20261029/"
        "installed-adapters/wheel_identity.py"
    )["verify_wheel"]
    root = tmp_path / "site-packages"
    wheel = tmp_path / "fixture.whl"
    with ZipFile(wheel, "w") as archive:
        for name in ("fixture/__init__.py", "fixture/basic.py"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            archive.writestr(name, b"")
    dist = SimpleNamespace(version="1", locate_file=root.joinpath)
    monkeypatch.setitem(verify.__globals__, "distribution", lambda _: dist)
    module = SimpleNamespace(__file__=str(root / "fixture/__init__.py"))
    monkeypatch.setitem(verify.__globals__, "import_module", lambda _: module)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    digest = sha256(wheel.read_bytes()).hexdigest()
    assert verify("fixture", "fixture", "1", digest, wheel) == 2
    extra = root / "fixture/basic/__init__.py"
    extra.parent.mkdir()
    extra.touch()
    with pytest.raises(ValueError, match="Unreviewed installed file"):
        verify("fixture", "fixture", "1", digest, wheel)
