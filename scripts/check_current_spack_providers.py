"""Qualify current provider source declarations without claiming native builds."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tomllib
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def _requirements(values: list[str]) -> set[Requirement]:
    return {Requirement(value) for value in values}


def _recipe(path: Path) -> dict[str, Any]:
    """Inspect declaration literals without executing downloaded or recipe code."""
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
    versions = []
    dependencies = {}
    licenses = []
    environment = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "version":
                versions.append(
                    (
                        ast.literal_eval(node.args[0]),
                        {k.arg: ast.literal_eval(k.value) for k in node.keywords},
                    )
                )
            elif name == "license":
                licenses.append(ast.literal_eval(node.args[0]))
            elif name == "depends_on":
                spec = ast.literal_eval(node.args[0])
                if spec in dependencies:
                    raise ValueError("Duplicate provider dependency")
                options = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                if set(options) != {"type"}:
                    raise ValueError("Conditional or implicit provider dependency")
                kinds = options["type"]
                dependencies[spec] = sorted(
                    [kinds] if isinstance(kinds, str) else kinds
                )
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
        ):
            if node.func.attr != "set" or len(node.args) != 2 or node.keywords:
                raise ValueError("Provider changes upstream build defaults")
            key = ast.literal_eval(node.args[0])
            if key not in {"RUSTC", "CARGO"} or key in environment:
                raise ValueError("Provider changes upstream build defaults")
            environment[key] = ast.dump(node.args[1])
    return {
        "versions": versions,
        "dependencies": dependencies,
        "licenses": licenses,
        "environment": environment,
    }


def _source(archive: Path, row: dict[str, Any]) -> dict[str, bytes]:
    """Authenticate an upstream source archive and read only bounded metadata."""
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != row["source"]["sha256"]:
        raise ValueError("Provider source checksum mismatch")
    selected = {"pyproject.toml", "LICENSE"}
    if "maturin_manifest" in row:
        selected.update({row["maturin_manifest"], "rust-toolchain.toml"})
    contents = {}
    with tarfile.open(archive, "r|gz", stream=True) as package:
        for member in package:
            name = member.name.split("/", 1)[-1]
            if name not in selected:
                continue
            if name in contents or not member.isfile() or member.size > 2 * 1024**2:
                raise ValueError("Invalid provider source metadata")
            stream = package.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable provider source metadata")
            contents[name] = stream.read()
    if contents.keys() != selected:
        raise ValueError("Missing provider source metadata")
    return contents


def validate(root: Path, sources: Path) -> dict[str, object]:
    """Check exact upstream archives, core dependencies, compiler and features."""
    packet = json.loads(
        (root / "packaging/current/polars-provider-sources.json").read_text()
    )
    if set(packet["providers"]) != {"py-polars", "py-polars-runtime-32"}:
        raise ValueError("Both split Polars providers are required")
    if packet["python_profile"] != "3.15.1":
        raise ValueError("Provider Python profile must be 3.15.1")
    for name, row in packet["providers"].items():
        recipe = _recipe(root / f"packaging/current/spack/packages/{name}/package.py")
        expected = (
            row["version"],
            {"sha256": row["source"]["sha256"], "url": row["source"]["url"]},
        )
        if recipe["versions"] != [expected] or recipe["licenses"] != [row["license"]]:
            raise ValueError("Provider identity or license drift")
        if recipe["dependencies"] != {
            key: sorted(value) for key, value in row["spack_dependencies"].items()
        }:
            raise ValueError("Provider core dependency drift")
        contents = _source(sources / row["source"]["filename"], row)
        configuration = tomllib.loads(contents["pyproject.toml"].decode())
        project = configuration["project"]
        if (
            project["name"] != name.removeprefix("py-")
            or Version(packet["python_profile"])
            not in SpecifierSet(project["requires-python"])
            or project["requires-python"] != row["requires_python"]
            or _requirements(project.get("dependencies", []))
            != _requirements(row["run_requires"])
            or _requirements(configuration["build-system"]["requires"])
            != _requirements(row["build_requires"])
            or configuration["build-system"]["build-backend"] != row["build_backend"]
            or project["license"] != {"file": "LICENSE"}
            or hashlib.sha256(contents["LICENSE"]).hexdigest() != row["license_sha256"]
        ):
            raise ValueError("Provider source contract drift")
        if "maturin_manifest" in row:
            manifest = row["maturin_manifest"]
            cargo = tomllib.loads(contents[manifest].decode())
            channel = tomllib.loads(contents["rust-toolchain.toml"].decode())[
                "toolchain"
            ]["channel"]
            expected_environment = {
                key: ast.dump(
                    ast.parse(
                        f'str(self.spec["rust"].prefix.bin.{binary})', mode="eval"
                    ).body
                )
                for key, binary in (("RUSTC", "rustc"), ("CARGO", "cargo"))
            }
            if (
                configuration["tool"]["maturin"]["manifest-path"] != manifest
                or channel != row["rust_toolchain"]
                or cargo["package"]["version"] != row["version"]
                or cargo["features"]["default"] != row["default_features"]
                or recipe["dependencies"].get("rust@=" + channel) != ["build"]
                or recipe["environment"] != expected_environment
            ):
                raise ValueError("Provider compiler or default features drift")
        elif (
            project["version"] != row["version"]
            or recipe["dependencies"].get("py-polars-runtime-32@=" + row["version"])
            != ["run"]
            or recipe["environment"]
        ):
            raise ValueError("Provider frontend version or runtime binding drift")
    return {
        "source_declarations_qualified": True,
        "providers_checked": len(packet["providers"]),
        "native_solver_executed": False,
        "native_build_executed": False,
        "installed_module_acceptance": False,
    }


def main() -> int:
    """Inspect supplied real archives; do not download or execute upstream code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1):
        parser.error("Provider qualification requires Python 3.15.1")
    try:
        report = validate(args.root, args.sources)
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
                {"source_declarations_qualified": False, "error": str(error)}, indent=2
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
