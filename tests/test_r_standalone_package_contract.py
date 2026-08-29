from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R_PACKAGE = ROOT / "r-package" / "voiageR"


def test_r_source_package_owns_its_native_build() -> None:
    description = (R_PACKAGE / "DESCRIPTION").read_text(encoding="utf-8")
    assert "NeedsCompilation: yes" in description
    assert "voiage-ffi shared library" not in description

    rust_manifest = R_PACKAGE / "src" / "rust" / "Cargo.toml"
    assert rust_manifest.is_file()
    manifest = rust_manifest.read_text(encoding="utf-8")
    assert 'crate-type = ["staticlib"]' in manifest
    assert "[dependencies]" not in manifest
    rust_source = (R_PACKAGE / "src" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "#![no_std]" in rust_source
    assert "use std::" not in rust_source
    assert "vec![" not in rust_source

    assert (R_PACKAGE / "src" / "Makevars").is_file()
    assert (R_PACKAGE / "src" / "Makevars.win").is_file()
    assert (R_PACKAGE / "src" / "init.c").is_file()


def test_r_runtime_never_loads_an_ambient_ffi_library() -> None:
    runtime = (R_PACKAGE / "R" / "voiageR.R").read_text(encoding="utf-8")
    assert "VOIAGE_FFI_LIBRARY" not in runtime
    assert "dyn.load" not in runtime
    assert ".C(\n    voiageR_evpi" in runtime
    assert ".C(\n    voiageR_enbs" in runtime


def test_r_native_tests_exercise_the_installed_package_without_environment_help() -> (
    None
):
    native_tests = (R_PACKAGE / "tests" / "testthat" / "test-native-ffi.R").read_text(
        encoding="utf-8"
    )
    assert "VOIAGE_FFI_LIBRARY" not in native_tests
    assert "skip_if" not in native_tests


def test_r_ci_installs_the_source_package_without_a_repository_ffi_library() -> None:
    workflow = (ROOT / ".github" / "workflows" / "bindings-ci.yml").read_text(
        encoding="utf-8"
    )
    r_jobs = workflow[workflow.index("  r:\n") :]
    assert "Build the Rust C ABI used by R" not in r_jobs
    assert "VOIAGE_FFI_LIBRARY" not in r_jobs
    assert "x86_64-pc-windows-gnu" in r_jobs
