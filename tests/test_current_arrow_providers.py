"""Source-contract controls for Python 3.15 Arrow build prerequisites."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts.check_current_arrow_providers import validate_libcst

ROOT = Path(__file__).parents[1]


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "packaging/current", root / "packaging/current")
    sources = tmp_path / "sources"
    sources.mkdir()
    path = root / "packaging/current/arrow-provider-sources.json"
    packet = json.loads(path.read_text())
    row = packet["providers"]["py-libcst"]
    files = {name: (name + " fixture license").encode() for name in row["licenses"]}
    files["pyproject.toml"] = (
        b'[project]\nname="libcst"\nrequires-python=">=3.9"\n'
        b"dependencies=[\"pyyaml>=5.2; python_version < '3.13'\","
        b"\"pyyaml-ft>=8.0.0; python_version == '3.13'\","
        b"\"pyyaml>=6.0.3; python_version >= '3.14'\","
        b"\"typing-extensions; python_version < '3.10'\"]\n"
        b'[project.license]\nfile="LICENSE"\n[build-system]\n'
        b'requires=["setuptools","setuptools-scm","setuptools-rust","wheel"]\n'
    )
    files["native/libcst/Cargo.toml"] = (
        b'[package]\nname="libcst"\nversion="1.9.0"\nrust-version="1.83"\n'
        b'license="MIT AND (MIT AND PSF-2.0)"\n'
        b'[features]\ndefault=["py"]\n[dependencies.pyo3]\nversion="0.29"\noptional=true\n'
    )
    row["licenses"] = {
        name: hashlib.sha256(files[name]).hexdigest() for name in row["licenses"]
    }
    archive = sources / row["source"]["filename"]
    with tarfile.open(archive, "w:gz") as output:
        for name, data in files.items():
            member = tarfile.TarInfo("libcst-1.9.0/" + name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    old = row["source"]["sha256"]
    row["source"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    recipe = root / "packaging/current/spack/packages/py-libcst/package.py"
    recipe.write_text(recipe.read_text().replace(old, row["source"]["sha256"]))
    path.write_text(json.dumps(packet))
    return root, sources


def test_source_profile_keeps_native_acceptance_separate(
    candidate: tuple[Path, Path],
) -> None:
    report = validate_libcst(*candidate)
    assert report["libcst_source_declaration_qualified"] is True
    assert report["python_profile"] == "3.15.1"
    assert report["spack_build_executed"] is False
    assert report["linux_build_executed"] is False
    assert report["full_arrow_profile_qualified"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "license",
        "runtime",
        "compiler",
        "features",
        "missing-license",
        "wrong-profile",
        "empty-providers",
    ],
)
def test_source_profile_rejects_unfaithful_declarations(
    candidate: tuple[Path, Path], damage: str
) -> None:
    root, sources = candidate
    recipe = root / "packaging/current/spack/packages/py-libcst/package.py"
    packet_path = root / "packaging/current/arrow-provider-sources.json"
    packet = json.loads(packet_path.read_text())
    if damage == "checksum":
        archive = next(sources.iterdir())
        archive.write_bytes(archive.read_bytes() + b"changed")
    elif damage == "license":
        recipe.write_text(recipe.read_text().replace("MIT AND PSF-2.0", "Apache-2.0"))
        packet["providers"]["py-libcst"]["license"] = "Apache-2.0"
    elif damage == "runtime":
        recipe.write_text(
            recipe.read_text().replace("py-pyyaml@6.0.3:", "py-pyyaml-ft@8:")
        )
    elif damage == "compiler":
        recipe.write_text(recipe.read_text().replace("rust@1.85:1", "rust@nightly"))
    elif damage == "features":
        recipe.write_text(
            recipe.read_text()
            + '        env.set("CARGO_BUILD_FEATURES", "--no-default-features")\n'
        )
    elif damage == "missing-license":
        packet["providers"]["py-libcst"]["licenses"].pop(
            "native/libcst/src/tokenizer/core/LICENSE"
        )
    elif damage == "wrong-profile":
        packet["python_profile"] = "3.14.0"
    elif damage == "empty-providers":
        packet["providers"] = {}
    packet_path.write_text(json.dumps(packet))
    with pytest.raises((ValueError, KeyError)):
        validate_libcst(root, sources)


def test_unknown_build_requirement_is_not_accepted_by_changing_the_packet(
    candidate: tuple[Path, Path],
) -> None:
    root, sources = candidate
    packet_path = root / "packaging/current/arrow-provider-sources.json"
    packet = json.loads(packet_path.read_text())
    row = packet["providers"]["py-libcst"]
    archive = sources / row["source"]["filename"]
    files = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            stream = source.extractfile(member)
            assert stream is not None
            files[member.name] = stream.read()
    name = "libcst-1.9.0/pyproject.toml"
    files[name] = files[name].replace(b'"wheel"]', b'"wheel","requests"]')
    with tarfile.open(archive, "w:gz") as output:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    old = row["source"]["sha256"]
    row["source"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    row["build_requires"].append("requests")
    packet_path.write_text(json.dumps(packet))
    recipe = root / "packaging/current/spack/packages/py-libcst/package.py"
    recipe.write_text(recipe.read_text().replace(old, row["source"]["sha256"]))
    with pytest.raises(ValueError):
        validate_libcst(root, sources)


@pytest.fixture
def cpp_candidate(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "packaging/current", root / "packaging/current")
    sources = tmp_path / "sources"
    sources.mkdir()
    packet_path = root / "packaging/current/arrow-provider-sources.json"
    packet = json.loads(packet_path.read_text())
    row = packet["providers"]["arrow"]
    files = {
        "LICENSE.txt": b"Apache fixture license",
        "NOTICE.txt": b"Apache fixture notice",
        "cpp/CMakeLists.txt": b'cmake_minimum_required(VERSION 3.25)\nset(ARROW_VERSION "26.0.0")\n',
        "cpp/cmake_modules/DefineOptions.cmake": b'define_option(ARROW_MIMALLOC "Build the Arrow mimalloc-based allocator" ON)\n',
        "cpp/thirdparty/versions.txt": b"ARROW_MIMALLOC_BUILD_VERSION=v3.4.1\nARROW_MIMALLOC_BUILD_SHA256_CHECKSUM=37107a52c16baa80c5f74861dddda7b27bb9949e41a6637691867a94c88ca446\n",
    }
    row["licenses"] = {
        name: hashlib.sha256(files[name]).hexdigest() for name in row["licenses"]
    }
    archive = sources / row["source"]["filename"]
    with tarfile.open(archive, "w:gz") as output:
        for name, data in files.items():
            member = tarfile.TarInfo("arrow-commit/" + name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    old = row["source"]["sha256"]
    row["source"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    recipe = root / "packaging/current/spack/packages/arrow/package.py"
    recipe.write_text(recipe.read_text().replace(old, row["source"]["sha256"]))
    builtin = tmp_path / "builtin-arrow.py"
    builtin.write_text("# pinned upstream recipe fixture\n")
    row["builtin_recipe_sha256"] = hashlib.sha256(builtin.read_bytes()).hexdigest()
    packet_path.write_text(json.dumps(packet))
    return root, sources, builtin


def test_cpp_source_and_allocator_are_bound_without_native_claims(
    cpp_candidate: tuple[Path, Path, Path],
) -> None:
    from scripts.check_current_arrow_providers import validate_arrow_cpp

    report = validate_arrow_cpp(*cpp_candidate)
    assert report["cpp_source_declaration_qualified"] is True
    assert report["allocator_default_preserved"] is True
    assert report["native_build_executed"] is False
    assert report["full_arrow_profile_qualified"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "resource",
        "url",
        "builtin",
        "source",
        "allocator-off",
        "base-binding",
        "allocator-override",
    ],
)
def test_cpp_profile_rejects_source_and_allocator_drift(
    cpp_candidate: tuple[Path, Path, Path], damage: str
) -> None:
    from scripts.check_current_arrow_providers import validate_arrow_cpp

    root, sources, builtin = cpp_candidate
    recipe = root / "packaging/current/spack/packages/arrow/package.py"
    if damage == "resource":
        recipe.write_text(
            recipe.read_text().replace(
                "37107a52c16baa80c5f74861dddda7b27bb9949e41a6637691867a94c88ca446",
                "0" * 64,
            )
        )
    elif damage == "url":
        recipe.write_text(
            recipe.read_text().replace(
                "446167169a0dc547e00e1bc29f417a40c0ca267a", "0" * 40
            )
        )
    elif damage == "builtin":
        builtin.write_text("# unreviewed catalogue\n")
    elif damage == "source":
        next(sources.iterdir()).write_bytes(b"changed")
    elif damage == "base-binding":
        recipe.write_text(
            recipe.read_text().replace(
                "builtin.packages.arrow.package", "builtin.packages.other.package"
            )
        )
    elif damage == "allocator-override":
        recipe.write_text(
            recipe.read_text()
            + '\n    def cmake_args(self):\n        return ["-DARROW_MIMALLOC=OFF"]\n'
        )
    elif damage == "allocator-off":
        recipe.write_text(recipe.read_text().replace("+mimalloc", "~mimalloc"))
    with pytest.raises(ValueError):
        validate_arrow_cpp(root, sources, builtin)


def test_allocator_hook_works_without_a_parent_package_hook(tmp_path: Path) -> None:
    import ast
    from types import SimpleNamespace

    tree = ast.parse(
        (ROOT / "packaging/current/spack/packages/arrow/package.py").read_text()
    )
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup_build_environment"
    )
    cls.body = [function]
    cls.bases = [ast.Name(id="ParentWithoutHook", ctx=ast.Load())]
    ast.fix_missing_locations(cls)
    namespace = {
        "ParentWithoutHook": type("ParentWithoutHook", (), {}),
        "Path": Path,
        "EnvironmentModifications": object,
    }
    exec(  # noqa: S102 - fixed repository hook AST; test-only parent and globals
        compile(ast.Module(body=[cls], type_ignores=[]), "allocator-hook", "exec"),
        namespace,
    )
    package = namespace["Arrow"]()
    package.spec = SimpleNamespace(
        satisfies=lambda constraint: constraint == "@=26.0.0+mimalloc"
    )
    package.stage = SimpleNamespace(source_path=str(tmp_path))
    archive = tmp_path / "spack-vendor/mimalloc-v3.4.1.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"source fixture")
    values = {}
    package.setup_build_environment(SimpleNamespace(set=values.__setitem__))
    assert values == {"ARROW_MIMALLOC_URL": archive.as_uri()}
    archive.unlink()
    with pytest.raises(RuntimeError, match="not staged"):
        package.setup_build_environment(SimpleNamespace(set=values.__setitem__))
