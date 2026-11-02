"""Compiler source proof must not accept bootstrap or downloaded output."""

lazy from pathlib import Path

import pytest

from scripts.qualify_native_rust import (
    bootstrap_config,
    validate_config,
    validate_installed_version,
)


def test_config_preserves_upstream_llvm_backends_and_disables_downloads(
    tmp_path: Path,
) -> None:
    config = bootstrap_config(tmp_path, "rust-stable")
    validate_config(config, tmp_path)
    assert "targets" not in config["llvm"]
    assert config["build"]["target"] == ["x86_64-unknown-linux-gnu"]
    assert config["build"]["locked-deps"]
    assert config["build"]["tools"] == ["cargo", "rustdoc", "src"]


@pytest.mark.parametrize(
    ("section", "key"), [("llvm", "download-ci-llvm"), ("rust", "download-rustc")]
)
def test_downloaded_compilers_cannot_be_accepted(
    tmp_path: Path, section: str, key: str
) -> None:
    config = bootstrap_config(tmp_path, "rust-stable")
    config[section][key] = True
    with pytest.raises(ValueError, match="source-built"):
        validate_config(config, tmp_path)


def test_bootstrap_is_not_the_qualified_compiler() -> None:
    with pytest.raises(ValueError, match="installed compiler"):
        validate_installed_version("rustc 1.98.0 (bootstrap)\n", "rust-stable")
    validate_installed_version(
        "rustc 1.99.1 (source)\nLLVM version: 22.1.0\n", "rust-stable"
    )
