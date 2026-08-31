"""Check dated compiler provenance and executable recipe configuration paths."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "packaging/spack-overlay"


def _method(package: str, method: str, namespace: dict[str, Any]) -> Any:
    """Execute the actual recipe method without installing Spack in test lanes."""
    tree = ast.parse((ROOT / "packages" / package / "package.py").read_text())
    node = next(
        child
        for parent in tree.body
        if isinstance(parent, ast.ClassDef)
        for child in parent.body
        if isinstance(child, ast.FunctionDef) and child.name == method
    )
    module = ast.Module(body=[node], type_ignores=[])
    methods: dict[str, Any] = {}
    exec(compile(module, f"{package}/package.py", "exec"), namespace, methods)  # noqa: S102 - executes checked-in recipe code
    return methods[method]


def test_dated_source_and_stage0_are_bound_to_official_metadata() -> None:
    audit = json.loads((ROOT / "rust-source-audit.json").read_text())
    compiler = audit["compiler"]
    stage0 = audit["stage0"]
    assert compiler["archive_digest_verified"]
    assert compiler["archive_downloaded"]
    assert compiler["bytes"] > 0
    assert compiler["sha256"] in (ROOT / "packages/rust/package.py").read_text()
    assert compiler["manifest_matches_official_sha256_sidecar"]
    assert stage0["matches_source_commit_file"]
    assert stage0["manifest_matches_official_sha256_sidecar"]
    assert stage0["channel"] == "beta"
    assert stage0["date"] == "2026-03-05"
    for target, entry in stage0["targets"].items():
        assert (
            entry["published_sha256"]
            in (ROOT / "packages/rust-bootstrap/package.py").read_text()
        )
        assert f"/2026-03-05/rust-beta-{target}.tar.xz" in entry["url"]
        assert not entry["installed"]
    assert not audit["compiler_installed"]
    assert not audit["native_builds_executed"]


def test_dated_urls_do_not_fall_through_generic_version_parser() -> None:
    rust_url = _method("rust", "url_for_version", {})
    assert rust_url(None, "nightly-2026-04-01") == (
        "https://static.rust-lang.org/dist/2026-04-01/rustc-nightly-src.tar.xz"
    )
    bootstrap_url = _method("rust-bootstrap", "url_for_version", {})
    for os_name, os_suffix in [
        ("darwin", "apple-darwin"),
        ("linux", "unknown-linux-gnu"),
    ]:
        for arch in ["x86_64", "aarch64"]:
            package = SimpleNamespace(
                os=os_name, target=arch, rust_os={os_name: os_suffix}
            )
            assert bootstrap_url(package, "beta-2026-03-05") == (
                "https://static.rust-lang.org/dist/2026-03-05/"
                f"rust-beta-{arch}-{os_suffix}.tar.xz"
            )


def test_dated_configure_uses_nightly_channel_and_explicit_bootstrap() -> None:
    calls: list[tuple[str, ...]] = []
    configure = _method(
        "rust", "configure", {"configure": lambda *args: calls.append(args)}
    )

    class Spec:
        def __init__(self, nightly: bool) -> None:
            self.nightly = nightly

        def satisfies(self, condition: str) -> bool:
            return self.nightly and condition == "@=nightly-2026-04-01"

        def __getitem__(self, name: str) -> Any:
            assert name == "rust-bootstrap"
            return SimpleNamespace(
                prefix=SimpleNamespace(
                    bin=SimpleNamespace(
                        cargo="/bootstrap/bin/cargo", rustc="/bootstrap/bin/rustc"
                    )
                )
            )

    for nightly in [True, False]:
        spec = Spec(nightly)
        configure(SimpleNamespace(spec=spec), spec, "/candidate")
        flags = calls[-1]
        expected = "nightly" if nightly else "stable"
        assert f"--release-channel={expected}" in flags
        assert "build.cargo=/bootstrap/bin/cargo" in flags
        assert "build.rustc=/bootstrap/bin/rustc" in flags
        assert "llvm.download-ci-llvm=false" in flags
        assert "build.vendor=true" in flags
        assert ("rust.download-rustc=false" in flags) is nightly


def test_bootstrap_install_passes_distinct_arguments() -> None:
    calls: list[tuple[str, ...]] = []
    install = _method(
        "rust-bootstrap",
        "install",
        {"Executable": lambda _: lambda *args: calls.append(args)},
    )
    install(None, None, "/prefix with spaces")
    assert calls == [("--prefix=/prefix with spaces", "--without=rust-docs")]


def test_polars_build_uses_its_direct_dated_rust_prefix() -> None:
    values: dict[str, str] = {}
    method = _method(
        "py-polars-runtime-32",
        "setup_build_environment",
        {"EnvironmentModifications": Any},
    )
    rust = SimpleNamespace(
        prefix=SimpleNamespace(
            bin=SimpleNamespace(rustc="/dated/bin/rustc", cargo="/dated/bin/cargo")
        )
    )
    method(
        SimpleNamespace(spec={"rust": rust}), SimpleNamespace(set=values.__setitem__)
    )
    assert values == {"RUSTC": "/dated/bin/rustc", "CARGO": "/dated/bin/cargo"}
    assert "RUSTC_BOOTSTRAP" not in values


def test_concrete_graph_keeps_only_the_four_build_tools_separate() -> None:
    receipt = json.loads((ROOT / "solver-receipt.json").read_text())
    artifact = receipt["concrete_dag"]
    path = ROOT / artifact["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    nodes = json.loads(path.read_text())["spec"]["nodes"]
    by_hash = {node["hash"]: node for node in nodes}
    counts = Counter(node["name"] for node in nodes)
    assert {name: count for name, count in counts.items() if count > 1} == {
        "rust": 2,
        "rust-bootstrap": 2,
        "py-maturin": 2,
        "py-setuptools-rust": 2,
    }
    for node in nodes:
        if node["name"] in {"py-maturin", "py-setuptools-rust"}:
            continue
        for edge in node.get("dependencies", []):
            if edge["name"] == "rust":
                assert edge["parameters"]["deptypes"] == ["build"]
                expected = (
                    "nightly-2026-04-01"
                    if node["name"] == "py-polars-runtime-32"
                    else "1.96.0"
                )
                assert by_hash[edge["hash"]]["version"] == expected
    for name, compiler in [
        ("py-polars-runtime-32", "nightly-2026-04-01"),
        ("py-pydantic-core", "1.96.0"),
    ]:
        node = next(node for node in nodes if node["name"] == name)
        edges = {edge["name"]: edge for edge in node["dependencies"]}
        assert by_hash[edges["rust"]["hash"]]["version"] == compiler
        assert edges["rust"]["parameters"]["deptypes"] == ["build"]
        maturin = by_hash[edges["py-maturin"]["hash"]]
        edges = {edge["name"]: edge for edge in maturin["dependencies"]}
        assert by_hash[edges["rust"]["hash"]]["version"] == compiler
        assert edges["rust"]["parameters"]["deptypes"] == ["build", "run"]
        plugin = by_hash[edges["py-setuptools-rust"]["hash"]]
        edge = next(edge for edge in plugin["dependencies"] if edge["name"] == "rust")
        assert by_hash[edge["hash"]]["version"] == compiler
        assert edge["parameters"]["deptypes"] == ["run"]


def test_solver_configuration_and_audits_are_manifest_bound() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name in [
        "concretizer.yaml",
        "rust-source-audit.json",
        "rust-backend-source-audit.json",
    ]:
        assert (
            manifest["support_sha256"][name]
            == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        )
    config = (ROOT / "concretizer.yaml").read_text()
    assert "unify: true" in config
    assert "strategy: minimal" in config
    assert "unify: false" not in config
    duplicate_limits = {
        line.split(":", 1)[0].strip(): int(line.split(":", 1)[1])
        for line in config.split("max_dupes:\n", 1)[1].splitlines()
        if line.strip()
    }
    assert duplicate_limits == {
        "rust": 2,
        "rust-bootstrap": 2,
        "py-maturin": 2,
        "py-setuptools-rust": 2,
    }


def test_intermediate_solver_failures_remain_inspectable() -> None:
    diagnostics = json.loads((ROOT / "dated-rust-diagnostics.json").read_text())
    assert [entry["exit_code"] for entry in diagnostics["attempts"]] == [0, 1, 1, 1]
    for entry in diagnostics["attempts"]:
        path = (ROOT / entry["path"]).resolve()
        assert path.is_relative_to((ROOT / "solver-logs").resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        if entry["exit_code"]:
            assert "Error: failed to concretize" in path.read_text()
