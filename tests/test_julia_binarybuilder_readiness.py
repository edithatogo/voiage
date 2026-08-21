"""Contracts for Julia artifact packaging and General registration."""

from pathlib import Path
import tomllib

import yaml

ROOT = Path(__file__).parents[1]


def test_binarybuilder_recipe_pins_release_and_supported_products() -> None:
    recipe = (ROOT / "packaging/yggdrasil/V/voiage_ffi/build_tarballs.jl").read_text()

    assert 'name = "voiage_ffi"' in recipe
    assert 'version = v"2.1.0"' in recipe
    assert "964a0fc334ece9509387cd07d43776adf38be240" in recipe
    assert 'LibraryProduct("libvoiage_ffi", :libvoiage_ffi)' in recipe
    assert "compilers = [:c, :rust]" in recipe
    assert "platforms = supported_platforms()" in recipe
    assert (
        'filter!(p -> !(Sys.isfreebsd(p) && arch(p) == "aarch64"), platforms)' in recipe
    )
    assert 'filter!(p -> arch(p) != "riscv64", platforms)' in recipe
    assert 'RUSTFLAGS="-C target-feature=-crt-static"' in recipe
    assert "install_license ../LICENSE" in recipe


def test_julia_package_has_general_quality_metadata() -> None:
    project_path = ROOT / "bindings/julia/Project.toml"
    project = tomllib.loads(project_path.read_text())

    assert project["name"] == "Voiage"
    assert project["authors"] == ["Dylan Mordaunt <voiage@users.noreply.github.com>"]
    assert "Statistics" not in project.get("deps", {})
    assert "JSON" not in project.get("deps", {})
    assert {"Aqua", "JSON", "Test"} <= set(project["extras"])
    assert {"Aqua", "JSON", "Test"} <= set(project["targets"]["test"])
    assert project["compat"]["Aqua"] == "0.8"
    assert project["compat"]["Libdl"] == "1.10"
    assert project["compat"]["Test"] == "1.10"
    assert (ROOT / "bindings/julia/LICENSE").read_bytes() == (
        ROOT / "LICENSE"
    ).read_bytes()
    assert not (ROOT / "bindings/julia/Manifest.toml").exists()


def test_tagbot_is_configured_for_the_julia_subpackage() -> None:
    workflow_path = ROOT / ".github/workflows/julia-tagbot.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    rendered = workflow_path.read_text()

    assert "issue_comment" in workflow[True]
    assert "workflow_dispatch" in workflow[True]
    assert workflow["jobs"]["tagbot"]["if"] == "github.actor == 'JuliaTagBot'"
    assert "subdir: bindings/julia" in rendered
    assert "tag_prefix: julia" in rendered
    assert "JuliaRegistries/TagBot@6b7c22e7bc2b8f4d1c56b7199a63421cf2667ed1" in rendered


def test_julia_ci_covers_supported_runtimes_and_platforms() -> None:
    workflow_path = ROOT / ".github/workflows/bindings-ci.yml"
    rendered = workflow_path.read_text()

    for runtime in ("1.10", "1.11", "1.12"):
        assert f'- "{runtime}"' in rendered
    for runner in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert f"runner: {runner}" in rendered
    assert "matrix.os.library" in rendered


def test_registration_documentation_preserves_the_two_external_gates() -> None:
    readme = (ROOT / "bindings/julia/README.md").read_text()

    assert "JuliaPackaging/Yggdrasil/pull/14292" in readme
    assert "@JuliaRegistrator register subdir=bindings/julia" in readme
    assert "JLL registration" in readme
    assert "General registry merge" in readme
