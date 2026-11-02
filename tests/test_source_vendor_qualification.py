"""Negative controls for authenticated source and vendor acceptance."""

from __future__ import annotations

import hashlib
import json
import tarfile
lazy from pathlib import Path

import pytest

from scripts.qualify_source_vendor import (
    compiler_vendor_contract,
    fetch,
    qualified_tool,
    unpack,
    validate_vendor,
    workspace_locks,
)


def test_member_manifests_reuse_the_authenticated_workspace_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Cargo.lock").write_text("frozen bytes")

    def locate(command, cwd, env, output):
        assert "--workspace" in command
        assert env["CARGO_NET_OFFLINE"] == "true"
        output.write_text(json.dumps({"root": str(source / "Cargo.toml")}))

    monkeypatch.setattr("scripts.qualify_source_vendor.run", locate)
    manifests = [source / "Cargo.toml", source / "src/tools/opt-dist/Cargo.toml"]
    assert workspace_locks(
        source, manifests, Path("/qualified/cargo"), {}, tmp_path
    ) == [source / "Cargo.lock"]
    assert (source / "Cargo.lock").read_text() == "frozen bytes"


def test_external_workspace_lock_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "Cargo.lock").write_text("external")

    def locate(command, cwd, env, output):
        output.write_text(json.dumps({"root": str(tmp_path / "Cargo.toml")}))

    monkeypatch.setattr("scripts.qualify_source_vendor.run", locate)
    with pytest.raises(ValueError, match="external authenticated workspace lock"):
        workspace_locks(
            source, [source / "Cargo.toml"], Path("/qualified/cargo"), {}, tmp_path
        )


def test_compiler_vendor_contract_covers_upstream_notice_workspaces(
    tmp_path: Path,
) -> None:
    module = tmp_path / "src/bootstrap/src/core/build_steps/vendor.rs"
    module.parent.mkdir(parents=True)
    module.write_text(
        "pub fn default_paths_to_vendor() {\n"
        '    [("library/Cargo.toml", []), ("src/tools/clippy/clippy_test_deps/Cargo.toml", [])]\n'
        '}\nfn unrelated() { "ignored/Cargo.toml" }\n'
    )
    with pytest.raises(ValueError, match="clippy_test_deps"):
        compiler_vendor_contract(
            tmp_path, [tmp_path / "Cargo.toml", tmp_path / "library/Cargo.toml"]
        )
    manifests = [
        tmp_path / p
        for p in (
            "Cargo.toml",
            "library/Cargo.toml",
            "src/tools/clippy/clippy_test_deps/Cargo.toml",
        )
    ]
    result = compiler_vendor_contract(tmp_path, manifests)
    assert result["required_manifests"] == sorted(
        str(p.relative_to(tmp_path)) for p in manifests
    )
    assert (
        result["upstream_module_sha256"]
        == hashlib.sha256(module.read_bytes()).hexdigest()
    )


def test_compiler_vendor_contract_rejects_an_unrecognized_upstream_layout(
    tmp_path: Path,
) -> None:
    module = tmp_path / "src/bootstrap/src/core/build_steps/vendor.rs"
    module.parent.mkdir(parents=True)
    module.write_text('fn unrelated() { "other/Cargo.toml" }\n')
    with pytest.raises(ValueError, match="upstream compiler vendor contract"):
        compiler_vendor_contract(tmp_path, [tmp_path / "Cargo.toml"])


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


def test_lock_repair_rejects_changed_input(tmp_path: Path) -> None:
    from scripts.qualify_source_vendor import apply_lock_patch

    lock = tmp_path / "Cargo.lock"
    lock.write_text("version = 4\n")
    patch = tmp_path / "lock.patch"
    patch.write_text("untrusted")
    with pytest.raises(ValueError, match="input digest mismatch"):
        apply_lock_patch(
            tmp_path,
            {"path": str(patch), "sha256": "a" * 64, "original_sha256": "b" * 64},
            "/usr/bin/git",
        )


def test_lock_repair_rejects_new_identity(tmp_path: Path) -> None:
    import difflib
    import shutil

    from scripts.qualify_source_vendor import apply_lock_patch

    before = 'version = 4\n\n[[package]]\nname = "example"\nversion = "1.0.0"\n'
    after = before.replace("1.0.0", "2.0.0")
    lock = tmp_path / "Cargo.lock"
    lock.write_text(before)
    patch = tmp_path / "lock.patch"
    patch.write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/Cargo.lock",
                tofile="b/Cargo.lock",
            )
        )
    )
    git = shutil.which("git")
    assert git is not None
    with pytest.raises(ValueError, match="new package identity"):
        apply_lock_patch(
            tmp_path,
            {
                "path": str(patch),
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "qualified_sha256": hashlib.sha256(after.encode()).hexdigest(),
            },
            git,
        )


def test_lock_repair_accepts_only_frozen_pruning(tmp_path: Path) -> None:
    import difflib
    import shutil

    from scripts.qualify_source_vendor import apply_lock_patch

    before = 'version = 4\n\n[[package]]\nname = "unused"\nversion = "1.0.0"\n'
    after = "version = 4\npackage = []\n"
    (tmp_path / "Cargo.lock").write_text(before)
    patch = tmp_path / "lock.patch"
    patch.write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/Cargo.lock",
                tofile="b/Cargo.lock",
            )
        )
    )
    git = shutil.which("git")
    assert git is not None
    result = apply_lock_patch(
        tmp_path,
        {
            "path": str(patch),
            "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "qualified_sha256": hashlib.sha256(after.encode()).hexdigest(),
        },
        git,
    )
    assert result["original_packages"] == 1
    assert result["qualified_packages"] == 0


def test_current_native_recipe_checksums_follow_actual_fetch_order() -> None:
    from scripts.qualify_git_archives import recipe_values

    repository = Path(__file__).resolve().parents[1]
    inputs = json.loads(
        (repository / "packaging/current/source-vendor-inputs.json").read_text()
    )
    for entry in inputs["entries"].values():
        for recipe in entry.get("recipes", []):
            values = recipe_values(repository / recipe)
            expected = [entry["source"]["filename"]]
            for crate in values["crates"]:
                filename = f"{crate[0]}-{crate[1]}"
                filename += f"-{crate[3][:8]}.tar.xz" if len(crate) == 4 else ".tar.gz"
                expected.append(filename)
            if "lock_patch" in entry:
                expected.append(Path(entry["lock_patch"]["path"]).name)
            assert len(expected) == len(values["checksums"]), recipe
            for filename, checksums in zip(expected, values["checksums"], strict=True):
                assert list(checksums) == [filename], (recipe, filename)
                assert len(checksums[filename]) == 64


def test_recipe_reader_rejects_executable_crate_declarations(tmp_path: Path) -> None:
    from scripts.qualify_git_archives import recipe_values

    recipe = tmp_path / "unsafe.eb"
    recipe.write_text('crates = __import__("os").system("false")\n')
    with pytest.raises(ValueError):
        recipe_values(recipe)


def test_conflicting_vendor_source_identities_fail_closed(tmp_path: Path) -> None:
    vendor, locks = vendor_fixture(tmp_path)
    conflict = dict(
        locks[0]["package"][0], source="git+https://example.org/repo#" + "a" * 40
    )
    with pytest.raises(ValueError, match="Conflicting vendor source"):
        validate_vendor(vendor, [*locks, {"package": [conflict]}])


def test_synchronized_local_workspace_is_not_an_external_vendor_conflict(
    tmp_path: Path,
) -> None:
    vendor, locks = vendor_fixture(tmp_path)
    local = {"name": "example", "version": "1.0.0"}
    result = validate_vendor(vendor, [*locks, {"package": [local]}])
    assert result["registry_packages"] == 1
    assert result["git_packages"] == 0


def test_oversized_fetch_cleans_only_its_owned_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    monkeypatch.setattr("scripts.qualify_source_vendor.MAX_ARCHIVE_BYTES", 3)
    monkeypatch.setattr(
        "scripts.qualify_source_vendor.urlopen",
        lambda *args, **kwargs: io.BytesIO(b"oversized"),
    )
    source = {
        "filename": "source.tar.gz",
        "url": "https://example.org/source.tar.gz",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="byte limit"):
        fetch(source, tmp_path)
    assert not list(tmp_path.iterdir())
    partial = tmp_path / "source.tar.gz.partial"
    partial.write_bytes(b"existing foreign partial")
    with pytest.raises(FileExistsError):
        fetch(source, tmp_path)
    assert partial.read_bytes() == b"existing foreign partial"
