from __future__ import annotations

from pathlib import Path

import pytest

from voiage import versioning


def test_version_sync_launcher_prefers_repository_package() -> None:
    """The skip-install tox launcher must not import a stale cached package."""
    launcher = Path("scripts/validate_version_sync.py").read_text(encoding="utf-8")

    path_setup = launcher.index("sys.path.insert")
    package_import = launcher.index("from voiage.versioning import main")
    assert path_setup < package_import


def _write_versioned_repo(root: Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        """
[project]
dynamic = ["version"]
""".strip(),
        encoding="utf-8",
    )
    (root / "rust/crates/voiage-python").mkdir(parents=True)
    (root / "rust/Cargo.toml").write_text(
        '[workspace.package]\nversion = "' + version + '"\n', encoding="utf-8"
    )
    (root / "rust/crates/voiage-python/Cargo.toml").write_text(
        "[package]\nversion.workspace = true\n", encoding="utf-8"
    )
    (root / "bindings/julia").mkdir(parents=True)
    (root / "bindings/julia/Project.toml").write_text(
        'version = "' + version + '"',
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
    (tmp_path / "bindings/julia/Project.toml").write_text(
        'version = "0.1.0"',
        encoding="utf-8",
    )

    with pytest.raises(versioning.VersionSyncError, match="Julia"):
        versioning.validate_version_sync(tmp_path)

    assert versioning.main(["--repo-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "version synchronization failed" in captured.err
    assert "bindings/julia/Project.toml" in captured.err


def test_release_tag_must_exactly_match_authoritative_cargo_version(
    tmp_path: Path,
) -> None:
    _write_versioned_repo(tmp_path, "0.2.0")

    assert versioning.validate_release_tag("v0.2.0", tmp_path) == "0.2.0"
    with pytest.raises(versioning.VersionSyncError, match="release tag"):
        versioning.validate_release_tag("v0.2.1", tmp_path)
    with pytest.raises(versioning.VersionSyncError, match="must match .*v0.2.0"):
        versioning.validate_release_tag("0.2.0", tmp_path)


def test_production_python_adapter_must_inherit_workspace_version(
    tmp_path: Path,
) -> None:
    _write_versioned_repo(tmp_path, "0.2.0")
    (tmp_path / "rust/crates/voiage-python/Cargo.toml").write_text(
        '[package]\nversion = "0.1.0"\n', encoding="utf-8"
    )

    with pytest.raises(versioning.VersionSyncError, match="must inherit"):
        versioning.validate_version_sync(tmp_path)

    (tmp_path / "rust/crates/voiage-python/Cargo.toml").write_text(
        '[package]\nversion = "0.2.0"\n', encoding="utf-8"
    )
    with pytest.raises(versioning.VersionSyncError, match="must inherit"):
        versioning.validate_version_sync(tmp_path)
