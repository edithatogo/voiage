"""Qualify current Arrow prerequisite sources without claiming native builds."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from scripts.check_current_spack_providers import _recipe, _requirements

LICENSES = {
    "LICENSE",
    "native/libcst/LICENSE",
    "native/libcst/src/tokenizer/core/LICENSE",
    "native/libcst_derive/LICENSE",
}


def validate_libcst(root: Path, sources: Path) -> dict[str, object]:
    """Bind the Python 3.15 source, native parser, licenses and stable compiler."""
    packet = json.loads(
        (root / "packaging/current/arrow-provider-sources.json").read_text()
    )
    if packet["python_profile"] != "3.15.1" or "py-libcst" not in packet["providers"]:
        raise ValueError("The Python 3.15.1 LibCST prerequisite is required")
    row = packet["providers"]["py-libcst"]
    recipe = _recipe(root / "packaging/current/spack/packages/py-libcst/package.py")
    if row["license"] != "MIT AND PSF-2.0" or set(row["licenses"]) != LICENSES:
        raise ValueError("LibCST native and upstream license coverage is required")
    if recipe["versions"] != [
        (
            row["version"],
            {"sha256": row["source"]["sha256"], "url": row["source"]["url"]},
        )
    ] or recipe["licenses"] != [row["license"]]:
        raise ValueError("LibCST source identity or license drift")
    expected = {
        "python@3.15.1": ["build", "run"],
        "py-setuptools": ["build"],
        "py-setuptools-scm": ["build"],
        "py-setuptools-rust": ["build"],
        "py-wheel": ["build"],
        "rust@1.85:1": ["build"],
        "py-pyyaml@6.0.3:": ["run"],
    }
    if recipe["dependencies"] != expected or row["spack_dependencies"] != expected:
        raise ValueError("LibCST Python-profile or stable compiler dependency drift")
    expected_environment = {
        key: ast.dump(
            ast.parse(f'str(self.spec["rust"].prefix.bin.{binary})', mode="eval").body
        )
        for key, binary in (("RUSTC", "rustc"), ("CARGO", "cargo"))
    }
    if recipe["environment"] != expected_environment:
        raise ValueError("LibCST must retain compiler binding and default features")
    archive = sources / row["source"]["filename"]
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != row["source"]["sha256"]:
        raise ValueError("LibCST source checksum mismatch")
    selected = LICENSES | {"pyproject.toml", "native/libcst/Cargo.toml"}
    contents = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = member.name.split("/", 1)[-1]
            if name not in selected:
                continue
            if name in contents or not member.isfile() or member.size > 2 * 1024**2:
                raise ValueError("Invalid LibCST source metadata")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable LibCST source metadata")
            contents[name] = stream.read()
    if set(contents) != selected or any(
        hashlib.sha256(contents[name]).hexdigest() != digest
        for name, digest in row["licenses"].items()
    ):
        raise ValueError("Missing or changed LibCST metadata or license")
    project = tomllib.loads(contents["pyproject.toml"].decode())
    cargo = tomllib.loads(contents["native/libcst/Cargo.toml"].decode())
    requirements = [Requirement(text) for text in project["project"]["dependencies"]]
    selected_run = set()
    for requirement in requirements:
        if requirement.marker is None or requirement.marker.evaluate(
            {"python_version": "3.15", "python_full_version": "3.15.1"}
        ):
            requirement.marker = None
            selected_run.add(requirement)
    if (
        project["project"]["name"] != "libcst"
        or project["project"]["requires-python"] != row["requires_python"]
        or Version("3.15.1") not in SpecifierSet(row["requires_python"])
        or project["project"]["license"] != {"file": "LICENSE"}
        or set(project["build-system"]) != {"requires"}
        or _requirements(project["build-system"]["requires"])
        != _requirements(row["build_requires"])
        or _requirements(project["build-system"]["requires"])
        != _requirements(["setuptools", "setuptools-scm", "setuptools-rust", "wheel"])
        or selected_run != _requirements(row["selected_run_requires"])
        or selected_run != _requirements(["pyyaml>=6.0.3"])
        or cargo["package"]["name"] != "libcst"
        or cargo["package"]["version"] != row["version"]
        or cargo["package"]["license"] != "MIT AND (MIT AND PSF-2.0)"
        or cargo["package"]["license"] != row["cargo_license"]
        or cargo["package"]["rust-version"] != row["rust_version"]
        or Version(row["rust_version"]) > Version("1.85")
        or cargo["features"]["default"] != row["default_features"]
        or row["default_features"] != ["py"]
        or cargo["dependencies"]["pyo3"]["version"] != row["pyo3_version"]
    ):
        raise ValueError("LibCST upstream source contract drift")
    return {
        "libcst_source_declaration_qualified": True,
        "python_profile": "3.15.1",
        "source_archive_sha256": checksum,
        "licenses_checked": len(LICENSES),
        "spack_build_executed": False,
        "linux_build_executed": False,
        "full_arrow_profile_qualified": False,
    }


def validate_arrow_cpp(
    root: Path, sources: Path, builtin_recipe: Path
) -> dict[str, object]:
    """Check immutable C++ source and the upstream allocator resource contract."""
    packet = json.loads(
        (root / "packaging/current/arrow-provider-sources.json").read_text()
    )
    row = packet["providers"]["arrow"]
    if (
        hashlib.sha256(builtin_recipe.read_bytes()).hexdigest()
        != row["builtin_recipe_sha256"]
    ):
        raise ValueError("Pinned builtin Arrow recipe drift")
    tree = ast.parse(
        (root / "packaging/current/spack/packages/arrow/package.py").read_text()
    )
    bindings = [
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == "BuiltinArrow"
    ]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if (
        bindings != [("spack_repo.builtin.packages.arrow.package", "Arrow")]
        or len(classes) != 1
        or [ast.dump(base) for base in classes[0].bases]
        != [ast.dump(ast.Name(id="BuiltinArrow", ctx=ast.Load()))]
        or any(
            isinstance(node, (ast.Assign, ast.AnnAssign)) for node in classes[0].body
        )
        or [node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)]
        != ["cmake_args", "setup_build_environment"]
    ):
        raise ValueError("Arrow must preserve the pinned upstream build implementation")
    calls = [
        (node.func.id, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    versions = [
        (
            ast.literal_eval(node.args[0]),
            {item.arg: ast.literal_eval(item.value) for item in node.keywords},
        )
        for name, node in calls
        if name == "version"
    ]
    expected = (
        row["version"],
        {"url": row["source"]["url"], "sha256": row["source"]["sha256"]},
    )
    if (
        versions != [expected]
        or row["source"]["url"]
        != f"https://github.com/apache/arrow/archive/{row['upstream_commit']}.tar.gz"
    ):
        raise ValueError("Arrow C++ immutable source identity drift")
    allocator = row["mimalloc"]
    resources = [
        {item.arg: ast.literal_eval(item.value) for item in node.keywords}
        for name, node in calls
        if name == "resource"
    ]
    if resources != [
        {
            "name": "mimalloc-3.4.1",
            "url": allocator["url"],
            "sha256": allocator["sha256"],
            "expand": False,
            "destination": "spack-vendor",
            "placement": {"v3.4.1.tar.gz": "mimalloc-v3.4.1.tar.gz"},
            "when": "@=26.0.0+mimalloc",
        }
    ]:
        raise ValueError("Arrow allocator source resource drift")
    archive = sources / row["source"]["filename"]
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != row["source"]["sha256"]:
        raise ValueError("Arrow C++ archive checksum mismatch")
    selected = {
        "LICENSE.txt",
        "NOTICE.txt",
        "cpp/CMakeLists.txt",
        "cpp/cmake_modules/DefineOptions.cmake",
        "cpp/thirdparty/versions.txt",
    }
    contents = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = member.name.split("/", 1)[-1]
            if name not in selected:
                continue
            if name in contents or not member.isfile() or member.size > 2 * 1024**2:
                raise ValueError("Invalid Arrow C++ metadata")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable Arrow C++ metadata")
            contents[name] = stream.read()
    if (
        set(contents) != selected
        or set(row["licenses"]) != {"LICENSE.txt", "NOTICE.txt"}
        or any(
            hashlib.sha256(contents[name]).hexdigest() != digest
            for name, digest in row["licenses"].items()
        )
    ):
        raise ValueError("Missing or changed Arrow C++ metadata/license")
    cmake = contents["cpp/CMakeLists.txt"].decode()
    versions = contents["cpp/thirdparty/versions.txt"].decode().splitlines()
    if (
        f'set(ARROW_VERSION "{row["version"]}")' not in cmake
        or f"cmake_minimum_required(VERSION {row['minimum_cmake']})" not in cmake
        or f"ARROW_MIMALLOC_BUILD_VERSION={allocator['version']}" not in versions
        or f"ARROW_MIMALLOC_BUILD_SHA256_CHECKSUM={allocator['sha256']}" not in versions
        or not re.search(
            r'define_option\(ARROW_MIMALLOC\s+"[^"]*"\s+ON\)',
            contents["cpp/cmake_modules/DefineOptions.cmake"].decode(),
        )
    ):
        raise ValueError("Arrow source version or default allocator contract drift")
    return {
        "cpp_source_declaration_qualified": True,
        "source_archive_sha256": checksum,
        "allocator_default_preserved": True,
        "allocator_archive_verified": False,
        "native_build_executed": False,
        "full_arrow_profile_qualified": False,
    }


ARROW_CORE_PROFILE = "arrow@=26.0.0+brotli+bz2+compute+csv+dataset+filesystem+ipc+lz4+mimalloc+parquet+python+shared+snappy+zlib+zstd"


def validate_pyarrow(root: Path, sources: Path) -> dict[str, object]:
    """Bind the frontend to its matching native library and supported I/O profile."""
    packet = json.loads(
        (root / "packaging/current/arrow-provider-sources.json").read_text()
    )
    row = packet["providers"]["py-pyarrow"]
    recipe_path = root / "packaging/current/spack/packages/py-pyarrow/package.py"
    recipe = _recipe(recipe_path)
    if (
        packet["python_profile"] != "3.15.1"
        or row["version"] != packet["providers"]["arrow"]["version"]
    ):
        raise ValueError("PyArrow must match the current Python and C++ profile")
    if (
        recipe["versions"]
        != [
            (
                row["version"],
                {"sha256": row["source"]["sha256"], "url": row["source"]["url"]},
            )
        ]
        or recipe["licenses"] != ["Apache-2.0"]
        or row["license"] != "Apache-2.0"
    ):
        raise ValueError("PyArrow source identity or license drift")
    expected = {
        "python@3.15.1": ["build", "run"],
        "c": ["build"],
        "cxx": ["build"],
        "cmake@3.25:": ["build"],
        "pkgconfig": ["build"],
        "py-scikit-build-core@1:": ["build"],
        "py-cython@3.1:": ["build"],
        "py-libcst@1.9:": ["build"],
        "py-numpy@2:": ["build"],
        "py-setuptools-scm@8:+toml": ["build"],
        ARROW_CORE_PROFILE: ["build", "link", "run"],
    }
    tree = ast.parse(recipe_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    modules = [
        ast.literal_eval(node.value)
        for node in cls.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "import_modules" for t in node.targets
        )
    ]
    if (
        row["native_profile"] != ARROW_CORE_PROFILE
        or recipe["dependencies"] != expected
        or recipe["environment"]
        or any(isinstance(node, ast.FunctionDef) for node in cls.body)
        or modules
        != [
            [
                "pyarrow",
                "pyarrow.compute",
                "pyarrow.csv",
                "pyarrow.dataset",
                "pyarrow.ipc",
                "pyarrow.parquet",
            ]
        ]
    ):
        raise ValueError(
            "PyArrow build prerequisites, native components or defaults drift"
        )
    archive = sources / row["source"]["filename"]
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != row["source"]["sha256"]:
        raise ValueError("PyArrow archive checksum mismatch")
    selected = {"LICENSE.txt", "NOTICE.txt", "pyproject.toml", "CMakeLists.txt"}
    if row["license_member_counts"] != {"LICENSE.txt": 2, "NOTICE.txt": 2}:
        raise ValueError(
            "The reviewed PyArrow license layout requires two identical copies"
        )
    expected_counts = {
        name: row["license_member_counts"].get(name, 1) for name in selected
    }
    counts = dict.fromkeys(selected, 0)
    contents = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = member.name.split("/", 1)[-1]
            if name not in selected:
                continue
            counts[name] += 1
            if (
                counts[name] > expected_counts[name]
                or not member.isfile()
                or member.size > 2 * 1024**2
            ):
                raise ValueError(
                    "Invalid or unexpected duplicate PyArrow source metadata"
                )
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable PyArrow source metadata")
            payload = stream.read()
            if name in contents and contents[name] != payload:
                raise ValueError("Changed duplicate PyArrow license metadata")
            contents[name] = payload
    if (
        set(contents) != selected
        or counts != expected_counts
        or set(row["licenses"]) != {"LICENSE.txt", "NOTICE.txt"}
        or any(
            hashlib.sha256(contents[name]).hexdigest() != digest
            for name, digest in row["licenses"].items()
        )
    ):
        raise ValueError("Missing or changed PyArrow source metadata/license")
    configuration = tomllib.loads(contents["pyproject.toml"].decode())
    project = configuration["project"]
    build = configuration["build-system"]
    selected_build = set()
    for text in build["requires"]:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate(
            {"python_version": "3.15", "python_full_version": "3.15.1"}
        ):
            requirement.marker = None
            selected_build.add(requirement)
    cmake = configuration["tool"]["scikit-build"]["cmake"]
    if (
        project["name"] != "pyarrow"
        or project.get("dependencies", [])
        or project["dynamic"] != ["version"]
        or project["license"] != "Apache-2.0"
        or set(project["license-files"]) != set(row["licenses"])
        or project["requires-python"] != row["requires_python"]
        or Version("3.15.1") not in SpecifierSet(row["requires_python"])
        or build["build-backend"] != "scikit_build_core.build"
        or row["build_backend"] != build["build-backend"]
        or _requirements(build["requires"]) != _requirements(row["build_requires"])
        or selected_build
        != _requirements(
            [
                "scikit-build-core>=1.0",
                "cython>=3.1",
                "libcst>=1.9.0",
                "numpy>=2.0",
                "setuptools-scm[toml]>=8",
            ]
        )
        or cmake["build-type"] != "Release"
        or row["cmake_build_type"] != "Release"
        or any(
            cmake["define"].get(key) != value or value != {"env": key, "default": "OFF"}
            for key, value in row["bundle_defaults"].items()
        )
        or set(row["bundle_defaults"])
        != {"PYARROW_BUNDLE_ARROW_CPP", "PYARROW_BUNDLE_CYTHON_CPP"}
        or configuration["tool"]["setuptools_scm"]["fallback_version"] != row["version"]
        or f"cmake_minimum_required(VERSION {row['minimum_cmake']})"
        not in contents["CMakeLists.txt"].decode()
    ):
        raise ValueError(
            "PyArrow upstream backend, requirements or default contract drift"
        )
    return {
        "pyarrow_source_declaration_qualified": True,
        "source_archive_sha256": checksum,
        "native_build_executed": False,
        "installed_module_acceptance": False,
    }


def validate_sources(
    root: Path, sources: Path, builtin_recipe: Path
) -> dict[str, object]:
    """Require every source layer; declaration checks never establish a native build."""
    packet = json.loads(
        (root / "packaging/current/arrow-provider-sources.json").read_text()
    )
    if set(packet["providers"]) != {"py-libcst", "arrow", "py-pyarrow"}:
        raise ValueError("All three current Arrow provider layers are required")
    reports = [
        validate_libcst(root, sources),
        validate_arrow_cpp(root, sources, builtin_recipe),
        validate_pyarrow(root, sources),
    ]
    return {
        "source_declarations_qualified": True,
        "providers_checked": len(reports),
        "qualification_scope": "Authenticated source/declarations only",
        "full_native_solver_executed": False,
        "native_build_executed": False,
        "installed_module_acceptance": False,
        "profiles": reports,
    }


def main() -> int:
    """Inspect supplied archives; never download, extract or execute source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile", choices=("libcst", "cpp", "pyarrow", "all"), default="libcst"
    )
    parser.add_argument("--builtin-arrow", type=Path)
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1):
        parser.error("Source qualification requires Python 3.15.1")
    try:
        if args.profile in {"cpp", "all"}:
            if args.builtin_arrow is None:
                parser.error("The C++ source profile requires --builtin-arrow")
            report = (
                validate_sources(args.root, args.sources, args.builtin_arrow)
                if args.profile == "all"
                else validate_arrow_cpp(args.root, args.sources, args.builtin_arrow)
            )
        elif args.profile == "pyarrow":
            report = validate_pyarrow(args.root, args.sources)
        else:
            report = validate_libcst(args.root, args.sources)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        SyntaxError,
        tarfile.TarError,
    ) as error:
        print(
            json.dumps(
                {
                    "source_declarations_qualified": False,
                    "profile": args.profile,
                    "error": str(error),
                }
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
