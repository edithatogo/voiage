"""Validate current overlay declarations and immutable historical provenance."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def historical_digest(path: Path) -> str:
    """Normalize text checkout newlines, never bytes inside compressed evidence."""
    raw = path.read_bytes()
    canonical = raw if b"\0" in raw[:8000] else raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def spack_requirement(text: str) -> str:
    """Translate the canonical supported bounds without broadening them."""
    requirement = Requirement(text)
    floor = next(s.version for s in requirement.specifier if s.operator == ">=")
    ceiling = next(
        (s.version for s in requirement.specifier if s.operator == "<"), None
    )
    high = ""
    if ceiling:
        parts = list(Version(ceiling).release)
        while len(parts) > 1 and parts[-1] == 0:
            parts.pop()
        parts[-1] -= 1
        high = ".".join(map(str, parts))
    return f"py-{canonicalize_name(requirement.name)}@{floor}:{high}"


def validate(root: Path) -> list[str]:
    """Check declarations only; never manufacture native build acceptance."""
    current = root / "packaging/current"
    canonical = json.loads((root / "specs/dependency-runtime-policy.json").read_text())
    runtime = json.loads((current / "runtime.json").read_text())
    errors = [
        f"Current overlay policy drift: {key}"
        for key in ("requires_python", "runtime")
        if runtime[key] != canonical[key]
    ]
    errors.extend(
        f"Unqualified native claim: {key}"
        for key in (
            "native_solver_executed",
            "native_build_executed",
            "installed_module_acceptance",
            "source_archive_qualified",
        )
        if runtime[key] is not False
    )
    lifecycle = json.loads((current / "overlay-lifecycle.json").read_text())
    for group in ("historical_files", "historical_tools"):
        for name, expected in lifecycle[group].items():
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                errors.append(f"Unsafe historical path: {name}")
            elif not path.is_file() or historical_digest(path) != expected:
                errors.append(f"Historical bytes changed: {name}")
    spack = ast.parse((current / "spack/packages/py-voiage/package.py").read_text())
    dependencies = {
        node.args[0].value
        for node in ast.walk(spack)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "depends_on"
    }
    expected = {spack_requirement(text) for text in canonical["runtime"]} | {
        "python@3.15.1",
        "rust@1.85:",
        "py-maturin@1.9:1",
    }
    if dependencies != expected:
        errors.append("Current Spack dependency drift")
    commits = [
        keyword.value.value
        for node in ast.walk(spack)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "version"
        for keyword in node.keywords
        if keyword.arg == "commit"
    ]
    if commits != [runtime["source_commit"]]:
        errors.append("Current Spack source drift")
    easy = ast.parse(
        (current / "easybuild/voiage-candidate-Python-3.15.1.eb").read_text()
    )
    dependencies = next(
        ast.literal_eval(n.value)
        for n in easy.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "dependencies"
    )
    pins = {canonicalize_name(name): version for name, version in dependencies}
    pins["pyarrow"] = pins.pop("arrow")
    if pins.pop("python") != "3.15.1":
        errors.append("Current EasyBuild interpreter drift")
    requirements = {
        canonicalize_name(Requirement(text).name): Requirement(text)
        for text in canonical["runtime"]
    }
    if pins.keys() != requirements.keys() or any(
        pins[name] not in req.specifier
        for name, req in requirements.items()
        if name in pins
    ):
        errors.append("Current EasyBuild dependency drift")
    return errors


def main() -> int:
    """Print a bounded declaration-only qualification result."""
    errors = validate(Path(__file__).resolve().parents[1])
    print(
        json.dumps(
            {"passed": not errors, "errors": errors, "native_build_executed": False},
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
