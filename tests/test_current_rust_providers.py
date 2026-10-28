"""Exercise dated compiler configuration without claiming a compiler build."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from packaging.version import Version
import pytest

ROOT = Path(__file__).parents[1]
PACKAGES = ROOT / "packaging/current/spack/packages"


def method(package: str, name: str, context: dict[str, object]) -> Callable[..., Any]:
    """Load one reviewed recipe method without executing Spack declarations."""
    tree = ast.parse((PACKAGES / package / "package.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    node.name = "_method_under_test"
    namespace = dict(context)
    exec(  # noqa: S102 - fixed repository method AST, captured test-only globals
        compile(ast.Module(body=[node], type_ignores=[]), "recipe-method", "exec"),
        namespace,
    )
    return cast("Callable[..., Any]", namespace["_method_under_test"])


@pytest.mark.parametrize("dated", [True, False])
def test_compiler_channel_bootstrap_and_source_construction_flags(dated: bool) -> None:
    commands = []
    function = method(
        "rust", "configure", {"configure": lambda *args: commands.extend(args)}
    )

    class Spec:
        def satisfies(self, constraint: str) -> bool:
            return constraint == "+src" or (
                dated and constraint == "@=nightly-2026-09-01"
            )

        def __getitem__(self, name: str) -> SimpleNamespace:
            assert name == "rust-bootstrap"
            return SimpleNamespace(
                prefix=SimpleNamespace(
                    bin=SimpleNamespace(
                        cargo="/bootstrap/cargo", rustc="/bootstrap/rustc"
                    )
                )
            )

    spec = Spec()
    function(SimpleNamespace(spec=spec), spec, "/compiler")
    assert "llvm.download-ci-llvm=false" in commands
    assert "build.vendor=true" in commands
    assert "build.cargo=/bootstrap/cargo" in commands
    assert "build.rustc=/bootstrap/rustc" in commands
    if dated:
        assert "rust.download-rustc=false" in commands
        assert "--release-channel=nightly" in commands
        assert "--release-channel=stable" not in commands
    else:
        assert "--release-channel=stable" in commands


def test_dated_source_url_cannot_fall_back_to_unversioned_nightly() -> None:
    function = method("rust", "url_for_version", {"Version": Version})
    assert (
        function(None, "nightly-2026-09-01")
        == "https://static.rust-lang.org/dist/2026-09-01/rustc-nightly-src.tar.xz"
    )
    assert (
        function(None, Version("1.97.1"))
        == "https://static.rust-lang.org/dist/rustc-1.97.1-src.tar.xz"
    )


@pytest.mark.parametrize("target", ["aarch64", "x86_64"])
def test_dated_bootstrap_url_uses_the_selected_platform(target: str) -> None:
    function = method("rust-bootstrap", "url_for_version", {"Version": Version})
    instance = SimpleNamespace(
        os="linux", target=target, rust_os={"linux": "unknown-linux-gnu"}
    )
    assert (
        function(instance, "beta-2026-08-18")
        == f"https://static.rust-lang.org/dist/2026-08-18/rust-beta-{target}-unknown-linux-gnu.tar.xz"
    )


def test_dated_bootstrap_dependency_is_not_the_unversioned_sdk() -> None:
    tree = ast.parse((PACKAGES / "rust/package.py").read_text())
    declarations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "depends_on"
        ):
            options = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
            declarations.append((ast.literal_eval(node.args[0]), options))
    assert (
        "rust-bootstrap@=beta-2026-08-18",
        {"type": "build", "when": "@=nightly-2026-09-01"},
    ) in declarations
    assert not any(
        spec == "rust-bootstrap@nightly" and opts.get("when") == "@nightly"
        for spec, opts in declarations
    )
    assert ("python@3.15.1", {"type": "build"}) in declarations


@pytest.fixture
def source_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Provide a small source archive with an authenticated bootstrap manifest."""
    import hashlib
    import io
    import json
    import shutil
    import tarfile

    root = tmp_path / "repo"
    shutil.copytree(ROOT / "packaging/current", root / "packaging/current")
    path = root / "packaging/current/rust-provider-sources.json"
    packet = json.loads(path.read_text())
    sdk = packet["bootstrap"]
    manifest = tmp_path / "bootstrap.toml"
    text = (
        'date="2026-08-18"\n[pkg.rust]\nversion='
        + json.dumps(sdk["compiler_version"])
        + "\ngit_commit_hash="
        + json.dumps(sdk["commit"])
        + "\n"
    )
    for system, hashes in sdk["hashes"].items():
        platform = {"linux": "unknown-linux-gnu", "darwin": "apple-darwin"}[system]
        for arch, digest in hashes.items():
            target = f"{arch}-{platform}"
            text += (
                f'[pkg.rust.target.{target}]\navailable=true\nxz_hash="{digest}"\n'
                f'xz_url="https://static.rust-lang.org/dist/2026-08-18/rust-beta-{target}.tar.xz"\n'
            )
    manifest.write_text(text)
    stage = (
        "compiler_channel_manifest_hash="
        + hashlib.sha256(manifest.read_bytes()).hexdigest()
        + "\ncompiler_date=2026-08-18\ncompiler_version=beta\ncompiler_git_commit_hash="
        + sdk["commit"]
        + "\n"
    ).encode()
    archive = tmp_path / "source.tar.xz"
    with tarfile.open(archive, "w:xz") as output:
        member = tarfile.TarInfo("rustc-nightly-src/src/stage0")
        member.size = len(stage)
        output.addfile(member, io.BytesIO(stage))
    old = packet["nightly"]["sha256"]
    packet["nightly"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    recipe = root / "packaging/current/spack/packages/rust/package.py"
    recipe.write_text(recipe.read_text().replace(old, packet["nightly"]["sha256"]))
    path.write_text(json.dumps(packet))
    return root, archive, manifest


def test_source_bootstrap_binding_does_not_claim_binary_or_build_proof(
    source_candidate: tuple[Path, Path, Path],
) -> None:
    from scripts.check_current_rust_sources import validate

    report = validate(*source_candidate)
    assert report["stage0_binding_qualified"] is True
    assert report["bootstrap_platforms_checked"] == 4
    assert report["bootstrap_binaries_verified"] is False
    assert report["native_build_executed"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "manifest",
        "identity",
        "recipe",
        "platform",
        "claim",
        "missing-platforms",
    ],
)
def test_source_bootstrap_binding_rejects_drift(
    source_candidate: tuple[Path, Path, Path],
    damage: str,
) -> None:
    import json

    from scripts.check_current_rust_sources import validate

    root, archive, manifest = source_candidate
    packet_path = root / "packaging/current/rust-provider-sources.json"
    packet = json.loads(packet_path.read_text())
    if damage == "source":
        archive.write_bytes(archive.read_bytes() + b"drift")
    elif damage == "manifest":
        manifest.write_text(manifest.read_text() + "# changed\n")
    elif damage == "missing-platforms":
        sdk_recipe = root / "packaging/current/spack/packages/rust-bootstrap/package.py"
        source = sdk_recipe.read_text()
        tree = ast.parse(source)
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "dated_beta" for t in n.targets)
        )
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1 : node.end_lineno] = ["    dated_beta = {}\n"]
        sdk_recipe.write_text("".join(lines))
        packet["bootstrap"]["hashes"] = {}
    elif damage == "identity":
        packet["bootstrap"]["commit"] = "0" * 40
    elif damage == "claim":
        packet["native_build_qualified"] = True
    elif damage == "platform":
        packet["bootstrap"]["hashes"]["linux"]["aarch64"] = "0" * 64
    else:
        recipe = root / "packaging/current/spack/packages/rust-bootstrap/package.py"
        recipe.write_text(
            recipe.read_text().replace(
                packet["bootstrap"]["hashes"]["linux"]["aarch64"], "0" * 64
            )
        )
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        validate(*source_candidate)
