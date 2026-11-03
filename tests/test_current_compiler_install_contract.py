"""Require explicit authenticated bootstrap inputs before installed HPC runs."""

import ast
import json
from pathlib import Path

import pytest

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
