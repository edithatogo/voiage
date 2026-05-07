from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from voiage import versioning

if TYPE_CHECKING:
    from pathlib import Path


def _write_versioned_repo(root: Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        """
[project]
version = "{version}"
""".strip().format(version=version),
        encoding="utf-8",
    )
    (root / "bindings/typescript").mkdir(parents=True)
    (root / "bindings/typescript/package.json").write_text(
        '{"version": "' + version + '"}',
        encoding="utf-8",
    )
    (root / "bindings/julia").mkdir(parents=True)
    (root / "bindings/julia/Project.toml").write_text(
        'version = "' + version + '"',
        encoding="utf-8",
    )
    (root / "bindings/rust").mkdir(parents=True)
    (root / "bindings/rust/Cargo.toml").write_text(
        '[package]\nversion = "' + version + '"\n',
        encoding="utf-8",
    )
    (root / "bindings/dotnet/src/Voiage.Core").mkdir(parents=True)
    (root / "bindings/dotnet/src/Voiage.Core/Voiage.Core.csproj").write_text(
        "<Project><PropertyGroup><Version>"
        + version
        + "</Version></PropertyGroup></Project>",
        encoding="utf-8",
    )
    (root / "r-package/voiageR").mkdir(parents=True)
    (root / "r-package/voiageR/DESCRIPTION").write_text(
        "Package: voiageR\nVersion: " + version + "\n",
        encoding="utf-8",
    )


def test_validate_version_sync_accepts_matching_manifests(tmp_path: Path) -> None:
    _write_versioned_repo(tmp_path, "0.2.0")

    canonical, mismatches = versioning.validate_version_sync(tmp_path)

    assert canonical == "0.2.0"
    assert mismatches == []
    assert versioning.main(["--repo-root", str(tmp_path)]) == 0


def test_validate_version_sync_reports_manifest_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_versioned_repo(tmp_path, "0.2.0")
    (tmp_path / "bindings/typescript/package.json").write_text(
        '{"version": "0.1.0"}',
        encoding="utf-8",
    )

    with pytest.raises(versioning.VersionSyncError, match="TypeScript"):
        versioning.validate_version_sync(tmp_path)

    assert versioning.main(["--repo-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "version synchronization failed" in captured.err
    assert "bindings/typescript/package.json" in captured.err
