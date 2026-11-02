"""Negative controls for authenticated source and vendor acceptance."""

from __future__ import annotations

import hashlib
import json
import tarfile
lazy from pathlib import Path

import pytest

from scripts.qualify_source_vendor import fetch, qualified_tool, unpack, validate_vendor


def test_offline_missing_source_never_opens_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("An offline source check attempted a network request")

    monkeypatch.setattr("scripts.qualify_source_vendor.urlopen", forbidden)
    with pytest.raises(ValueError, match="Forbidden fetch"):
        fetch(
            {
                "filename": "source.tar.gz",
                "url": "https://example.org/source.tar.gz",
                "sha256": "a" * 64,
            },
            tmp_path,
            offline=True,
        )


def test_corrupt_cached_source_is_preserved_and_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        fetch(
            {
                "filename": source.name,
                "url": "https://example.org/source.tar.gz",
                "sha256": "a" * 64,
            },
            tmp_path,
        )
    assert source.read_bytes() == b"corrupt"


def test_shadowed_source_and_tool_paths_are_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"source")
    (tmp_path / "source.tar.gz").symlink_to(original)
    with pytest.raises(ValueError, match="Shadowed"):
        fetch(
            {
                "filename": "source.tar.gz",
                "url": "https://example.org/source.tar.gz",
                "sha256": hashlib.sha256(b"source").hexdigest(),
            },
            tmp_path,
        )
    original.chmod(0o755)
    with pytest.raises(ValueError, match="Unqualified tool"):
        qualified_tool(original, tmp_path)


def test_parent_path_in_archive_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as package:
        package.addfile(tarfile.TarInfo("../escape"))
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()


def vendor_fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    directory = tmp_path / "vendor" / "example-1.0.0"
    directory.mkdir(parents=True)
    files = {
        "Cargo.toml": '[package]\nname = "example"\nversion = "1.0.0"\n',
        "lib.rs": "pub const VALUE: i32 = 7;\n",
    }
    for name, body in files.items():
        (directory / name).write_text(body)
    checks = {
        "package": "a" * 64,
        "files": {
            name: hashlib.sha256(body.encode()).hexdigest()
            for name, body in files.items()
        },
    }
    (directory / ".cargo-checksum.json").write_text(json.dumps(checks))
    locks = [
        {
            "package": [
                {
                    "name": "example",
                    "version": "1.0.0",
                    "source": "registry+https://github.com/rust-lang/crates.io-index",
                    "checksum": "a" * 64,
                }
            ]
        }
    ]
    return directory.parent, locks


def test_vendor_acceptance_checks_files_and_lock_identity(tmp_path: Path) -> None:
    vendor, locks = vendor_fixture(tmp_path)
    assert validate_vendor(vendor, locks)["registry_packages"] == 1
    (vendor / "example-1.0.0/lib.rs").write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_vendor(vendor, locks)


def test_missing_and_extra_vendor_files_fail_closed(tmp_path: Path) -> None:
    vendor, locks = vendor_fixture(tmp_path)
    (vendor / "example-1.0.0/extra").write_text("unexpected")
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_vendor(vendor, locks)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="Missing vendor packages"):
        validate_vendor(empty, locks)
