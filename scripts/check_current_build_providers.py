"""Check source identities and build contracts for current Python prerequisites."""

import argparse
import ast
import email.parser
import hashlib
import json
from pathlib import Path
import tarfile
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def _recipe(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    controls = (
        ast.If,
        ast.IfExp,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Match,
    )
    if len(classes) != 1 or any(
        isinstance(node, controls) for node in ast.walk(classes[0])
    ):
        raise ValueError("Conditional provider code requires native qualification")
    expected_hook = (
        "setup_dependent_build_environment"
        if path.parent.name == "py-cython"
        else "setup_build_environment"
    )
    hooks = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == expected_hook
    ]
    if len(hooks) != 1:
        raise ValueError("Required Spack environment hook missing")
    env_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "env"
    ]
    if any(node not in list(ast.walk(hooks[0])) for node in env_calls):
        raise ValueError("Environment setting outside required hook")
    versions, dependencies, licenses, environment = [], {}, [], {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {
            "version",
            "depends_on",
            "license",
        }:
            name = node.func.id
            value = ast.literal_eval(node.args[0])
            kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            if name == "version":
                versions.append((value, kwargs))
            elif name == "license":
                licenses.append(value)
            else:
                if value in dependencies or set(kwargs) != {"type"}:
                    raise ValueError("Invalid provider dependency")
                kinds = kwargs["type"]
                dependencies[value] = sorted(
                    [kinds] if isinstance(kinds, str) else kinds
                )
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
        ):
            if node.func.attr != "set" or len(node.args) != 2 or node.keywords:
                raise ValueError("Unexpected environment change")
            key = ast.literal_eval(node.args[0])
            if key in environment:
                raise ValueError("Duplicate environment setting")
            environment[key] = ast.dump(node.args[1])
    return {
        "versions": versions,
        "dependencies": dependencies,
        "licenses": licenses,
        "environment": environment,
    }


def _contents(archive: Path, row: dict, names: set[str]) -> dict[str, bytes]:
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != row["source"]["sha256"]:
            raise ValueError("Source digest mismatch")
    found = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name.split("/", 1)[-1]
            if name not in names:
                continue
            if name in found or not member.isfile() or member.size > 2 * 1024**2:
                raise ValueError("Invalid source metadata member")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable source metadata")
            found[name] = stream.read()
    if set(found) != names:
        raise ValueError("Missing source metadata")
    for name, digest in row["licenses"].items():
        if hashlib.sha256(found[name]).hexdigest() != digest:
            raise ValueError("License digest mismatch")
    return found


def validate(root: Path, sources: Path) -> dict[str, object]:
    """Check declarations and metadata; do not claim an installed native build."""
    packet = json.loads(
        (root / "packaging/current/runtime-build-provider-sources.json").read_text()
    )
    if packet["python_profile"] != "3.15.1" or set(packet["providers"]) != {
        "py-cython",
        "py-pydantic-core",
    }:
        raise ValueError("Wrong provider profile")
    deps = {
        "py-cython": {
            "python@3.15.1": ["build", "link", "run"],
            "c": ["build"],
            "cxx": ["build"],
            "py-setuptools@66:": ["build", "run"],
        },
        "py-pydantic-core": {
            "python@3.15.1": ["build", "run"],
            "rust@1.88:1": ["build"],
            "py-maturin@1.10:1": ["build"],
            "py-typing-extensions@4.14.1:": ["run"],
        },
    }
    identities = {
        "py-cython": ("3.3.0", "Apache-2.0", {"COPYING.txt", "LICENSE.txt"}),
        "py-pydantic-core": ("2.46.4", "MIT", {"LICENSE"}),
    }
    for name, row in packet["providers"].items():
        expected_version, expected_license, expected_files = identities[name]
        if (row["version"], row["license"], set(row["licenses"])) != (
            expected_version,
            expected_license,
            expected_files,
        ):
            raise ValueError("Provider identity or license coverage changed")
        recipe = _recipe(root / f"packaging/current/spack/packages/{name}/package.py")
        if (
            recipe["versions"]
            != [
                (
                    row["version"],
                    {"sha256": row["source"]["sha256"], "url": row["source"]["url"]},
                )
            ]
            or recipe["licenses"] != [row["license"]]
            or recipe["dependencies"] != deps[name]
        ):
            raise ValueError("Provider declarations changed")
        expected_env = (
            {"CYTHON_FORCE_REGEN": ast.dump(ast.Constant("1"))}
            if name == "py-cython"
            else {
                key: ast.dump(
                    ast.parse(
                        f'str(self.spec["rust"].prefix.bin.{binary})', mode="eval"
                    ).body
                )
                for key, binary in (("RUSTC", "rustc"), ("CARGO", "cargo"))
            }
        )
        if recipe["environment"] != expected_env:
            raise ValueError("Provider environment changed")
        files = set(row["licenses"]) | (
            {"PKG-INFO", "setup.py"}
            if name == "py-cython"
            else {"pyproject.toml", "Cargo.toml"}
        )
        data = _contents(sources / row["source"]["filename"], row, files)
        if name == "py-cython":
            meta = email.parser.BytesParser().parsebytes(data["PKG-INFO"])
            if (
                meta["Name"].lower() != "cython"
                or meta["Version"] != row["version"]
                or Version("3.15.1") not in SpecifierSet(meta["Requires-Python"])
                or "PYTHON_REQUIRES" not in data["setup.py"].decode()
            ):
                raise ValueError("Cython source contract changed")
        else:
            project = tomllib.loads(data["pyproject.toml"].decode())
            cargo = tomllib.loads(data["Cargo.toml"].decode())
            if (
                project["project"]["name"] != "pydantic_core"
                or project["project"]["license"] != "MIT"
                or {Requirement(x) for x in project["project"].get("dependencies", [])}
                != {Requirement("typing-extensions>=4.14.1")}
                or cargo["package"]["version"] != row["version"]
                or cargo["package"]["license"] != "MIT"
                or project["build-system"]["build-backend"] != "maturin"
                or {Requirement(x) for x in project["build-system"]["requires"]}
                != {Requirement("maturin>=1.10,<2")}
                or Version("3.15.1")
                not in SpecifierSet(project["project"]["requires-python"])
                or cargo["package"]["rust-version"] != "1.88"
                or cargo["dependencies"]["pyo3"]["version"] != "0.28"
                or cargo["profile"]["release"]
                != {"lto": "fat", "codegen-units": 1, "strip": True}
            ):
                raise ValueError("Pydantic core source contract changed")
    return {
        "source_declarations_qualified": True,
        "providers_checked": 2,
        "native_build_executed": False,
        "full_hpc_matrix_qualified": False,
    }


def main() -> None:
    """Validate source archives from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.sources), indent=2))


if __name__ == "__main__":
    main()
