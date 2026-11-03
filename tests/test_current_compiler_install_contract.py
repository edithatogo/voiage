"""Require explicit authenticated bootstrap inputs before installed HPC runs."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.easybuild_source_hooks import post_configure_hook

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("entry", "version"), [("rust-stable", "1.98.0"), ("rust-nightly", "1.99.0-beta.1")]
)
def test_easybuild_bootstrap_matches_qualified_sdk(entry: str, version: str) -> None:
    inputs = json.loads(
        (ROOT / "packaging/current/source-vendor-inputs.json").read_text()
    )
    sdk = inputs["entries"][entry]["bootstrap"]
    path = ROOT / "packaging/current/easybuild" / f"RustBootstrap-{version}.eb"
    strings = {
        node.value
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert sdk["url"] in strings
    assert sdk["sha256"] in strings
    assert "--components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu" in " ".join(
        strings
    )


@pytest.mark.parametrize(
    ("generation", "compiler"), [("2023a", "12.3.0"), ("2024a", "13.3.0")]
)
@pytest.mark.parametrize(
    ("name", "version", "bootstrap"),
    [("Rust", "1.99.1", "1.98.0"), ("Rust-nightly", "2026-09-01", "1.99.0-beta.1")],
)
def test_easybuild_compiler_uses_explicit_bootstrap_and_one_llvm_link(
    generation: str, compiler: str, name: str, version: str, bootstrap: str
) -> None:
    path = (
        ROOT
        / "packaging/current/easybuild"
        / generation
        / f"{name}-{version}-GCCcore-{compiler}.eb"
    )
    tree = ast.parse(path.read_text())
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
    }
    text = path.read_text()
    assert "llvm.link-jobs=1" in text
    assert "build.rustc=$EBROOTRUSTBOOTSTRAP/bin/rustc" in text
    assert "build.cargo=$EBROOTRUSTBOOTSTRAP/bin/cargo" in text
    assert any(
        isinstance(node, ast.Constant) and node.value == bootstrap
        for node in ast.walk(assignments["builddependencies"])
    )


@pytest.mark.parametrize("provider", ["rust", "rust-polars"])
def test_spack_compilers_bound_llvm_link_concurrency(provider: str) -> None:
    source = (
        ROOT / "packaging/current/spack/packages" / provider / "package.py"
    ).read_text()
    assert 'opts.append("llvm.link-jobs=1")' in source


@pytest.mark.parametrize("invalid", [None, "llvm", "rust", "vendor", "prefix"])
def test_effective_compiler_configuration_fails_closed(
    tmp_path: Path, monkeypatch, invalid
) -> None:
    prefix = str(tmp_path / "install")
    text = (
        "[llvm]\ndownload-ci-llvm=false\nlink-jobs=1\n[rust]\ndownload-rustc=false\n[build]\nvendor=true\n[install]\nprefix="
        + json.dumps(prefix)
        + "\n"
    )
    if invalid == "llvm":
        text = text.replace("download-ci-llvm=false", "download-ci-llvm=true")
    elif invalid == "rust":
        text = text.replace("download-rustc=false", "download-rustc=true")
    elif invalid == "vendor":
        text = text.replace("vendor=true", "vendor=false")
    elif invalid == "prefix":
        text = text.replace(json.dumps(prefix), json.dumps(str(tmp_path / "other")))
    (tmp_path / "bootstrap.toml").write_text(text)
    output = tmp_path / "receipts"
    monkeypatch.setenv("VOIAGE_COMPILER_RECEIPTS", str(output))
    app = SimpleNamespace(
        name="Rust",
        version="1.99.1",
        toolchain=SimpleNamespace(version="12.3.0"),
        cfg={"start_dir": str(tmp_path)},
        installdir=prefix,
    )
    if invalid:
        with pytest.raises(ValueError, match="configuration"):
            post_configure_hook(app)
        assert not list(output.glob("*.toml"))
    else:
        post_configure_hook(app)
        assert next(output.glob("*.toml")).read_text() == text


def test_spack_root_keeps_qualified_identity_without_git_metadata() -> None:
    receipt = json.loads(
        (
            ROOT
            / "conductor/tracks/remaining_backlog_delivery_20260831/source-qualification-20261028.json"
        ).read_text()
    )
    tree = ast.parse(
        (ROOT / "packaging/current/spack/packages/py-voiage/package.py").read_text()
    )
    values = {
        call.args[0].value: call.args[1].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "set"
        and len(call.args) == 2
    }
    assert (
        values["VOIAGE_SOURCE_REVISION"] == "3df3cb49e5c927075c0609a48a2535b4457424d4"
    )
    assert values["VOIAGE_SOURCE_TREE_GIT_OID"] == receipt["source_tree"]
    assert values["VOIAGE_SOURCE_CLEAN"] == "true"
