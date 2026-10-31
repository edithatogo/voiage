"""Validate generation-neutral EasyBuild root dependency/provider contracts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_BUILD_DEPENDENCIES = (("Rust", "1.96.0"), ("maturin", "1.13.1"))
EXPECTED_RUNTIME_DEPENDENCIES = (
    ("Python", "3.12.14", ""),
    ("Voiage-Python-support", "2.2.0", ""),
    ("SciPy-bundle", "2026.09", "-voiage-2.2.0"),
    ("Voiage-scientific-consumers", "2.2.0", ""),
    ("Arrow", "25.0.1", ""),
    ("polars", "1.42.1", ""),
    ("pydantic", "2.13.4", ""),
    ("jsonschema", "4.26.0", ""),
)
FORBIDDEN_DIRECT_MODULES = frozenset(
    {
        "Click",
        "NumPy",
        "SciPy",
        "pandas",
        "xarray",
        "scikit-learn",
        "typing_extensions",
        "typer",
        "PyArrow",
    }
)
EXPECTED_PROVIDERS = {
    "pydantic": ("rust", "2.13.4", "pydantic/2.13.4"),
    "jsonschema": ("rust", "4.26.0", "jsonschema/4.26.0"),
    "polars": ("polars", "1.42.1", "polars/1.42.1"),
    "pyarrow": ("arrow", "25.0.1", "Arrow/25.0.1"),
}
EXPECTED_ROBOT_MODULE_COUNT = 108
EXPECTED_2024A_BUILD_TOOL_EXCEPTION = {
    "module": "Python/3.12.3-GCCcore-13.3.0",
    "consumer": "Meson/1.4.0-GCCcore-13.3.0",
    "consumer_role": "builddependency",
    "transitive_consumer": "libpciaccess/0.18.1-GCCcore-13.3.0",
    "transitive_consumer_role": "builddependency",
}


def _robot_modules(text: str) -> list[str]:
    """Return the exact module names recorded by a retained robot log."""
    modules: list[str] = []
    marker = "(module: "
    for line in text.splitlines():
        if marker not in line or not line.endswith(")"):
            continue
        modules.append(line.split(marker, 1)[1][:-1])
    return modules


def validate_robot_modules(
    record: dict[str, Any], generation: str, modules: list[str]
) -> list[str]:
    """Bind receipt claims to the exact modules parsed from its robot log."""
    errors: list[str] = []
    if len(set(modules)) != len(modules):
        errors.append(f"{generation} robot log contains duplicate modules")
    python_modules = [module for module in modules if module.startswith("Python/")]
    if record.get("python_modules") != python_modules:
        errors.append(f"{generation} receipt Python modules do not match its log")
    runtime_python = record.get("runtime_python_module")
    if runtime_python not in modules:
        errors.append(f"{generation} runtime Python is missing from its robot log")
    if generation == "2024a":
        exception = record.get("build_tool_exception", {})
        for field in ("module", "consumer", "transitive_consumer"):
            module = exception.get(field) if isinstance(exception, dict) else None
            if module not in modules:
                errors.append(
                    f"{generation} build-tool exception {field} is missing from its robot log"
                )
    return errors


class RootGraphContractError(ValueError):
    """Raised when a root EasyBuild graph is stale or incompletely bound."""


def _assignments(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _repository_file(repository: Path, value: str) -> Path | None:
    """Resolve a declared repository path without allowing traversal or symlinks."""
    candidate = repository / value
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    if candidate.is_symlink() or not resolved.is_file():
        return None
    return resolved


def _runtime_tuple(item: object) -> tuple[str, str, str]:
    if not isinstance(item, (tuple, list)) or len(item) not in {2, 3}:
        raise RootGraphContractError(f"invalid dependency entry: {item!r}")
    if not all(isinstance(value, str) for value in item):
        raise RootGraphContractError(f"non-string dependency entry: {item!r}")
    name, version = item[:2]
    suffix = item[2] if len(item) == 3 else ""
    return name, version, suffix


def validate_root_recipe(path: Path, generation: str) -> list[str]:
    """Return all root recipe violations for one toolchain generation."""
    recipe = _assignments(path)
    errors: list[str] = []
    if recipe.get("toolchain") != {"name": "foss", "version": generation}:
        errors.append(f"toolchain must be foss/{generation}")

    build = tuple(
        _runtime_tuple(item)[:2] for item in recipe.get("builddependencies", [])
    )
    if build != EXPECTED_BUILD_DEPENDENCIES:
        errors.append(
            f"builddependencies must be {EXPECTED_BUILD_DEPENDENCIES!r}, got {build!r}"
        )

    runtime = tuple(_runtime_tuple(item) for item in recipe.get("dependencies", []))
    if runtime != EXPECTED_RUNTIME_DEPENDENCIES:
        errors.append(
            f"dependencies must be {EXPECTED_RUNTIME_DEPENDENCIES!r}, got {runtime!r}"
        )
    forbidden = sorted({name for name, _, _ in runtime} & FORBIDDEN_DIRECT_MODULES)
    if forbidden:
        errors.append(
            f"stale direct modules must be supplied by bundles: {forbidden!r}"
        )
    return errors


def validate_provider_registry(
    path: Path, generation: str, repository: Path, recipe_path: Path
) -> list[str]:
    """Return provider registry violations, including immutable manifest bindings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("generation", data.get("toolchain_generation")) != generation:
        errors.append(f"provider registry generation must be {generation}")
    overlays = data.get("external_provider_overlays", {})
    evidence = data.get("external_provider_evidence", {})
    recipe_relative = recipe_path.relative_to(repository).as_posix()
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    for provider, (fragment, version, dependency) in EXPECTED_PROVIDERS.items():
        overlay = overlays.get(provider)
        expected_overlay = f"packaging/easybuild-{generation}-{fragment}-overlay"
        if overlay != expected_overlay:
            errors.append(f"{provider} must map to the {generation} {fragment} overlay")
        record = evidence.get(provider)
        if not isinstance(record, dict):
            errors.append(f"{provider} must have immutable external provider evidence")
            continue
        manifest = record.get("manifest")
        expected_sha = record.get("manifest_sha256")
        if not isinstance(manifest, str) or not isinstance(expected_sha, str):
            errors.append(f"{provider} evidence must declare manifest path and SHA-256")
            continue
        manifest_path = _repository_file(repository, manifest)
        if manifest_path is None:
            errors.append(f"{provider} evidence manifest is missing: {manifest}")
        elif hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_sha:
            errors.append(f"{provider} evidence manifest SHA-256 does not match")
        if record.get("version") != version:
            errors.append(f"{provider} evidence version must be {version}")
        if record.get("consumer_dependency") != dependency:
            errors.append(f"{provider} consumer dependency must be {dependency}")
        if record.get("consumer_recipe") != recipe_relative:
            errors.append(f"{provider} evidence must bind the current root recipe path")
        if record.get("consumer_recipe_sha256") != recipe_sha:
            errors.append(
                f"{provider} evidence must bind the current root recipe SHA-256"
            )
        if record.get("native_build_executed") is not False:
            errors.append(f"{provider} native-build status must remain false")
        if record.get("full_voiage_ready") is not False:
            errors.append(f"{provider} full-graph status must remain false")
    return errors


def validate_generation(repository: Path, generation: str) -> list[str]:
    """Validate the root recipe and provider registry for a generation."""
    recipe = repository / f"packaging/easybuild/voiage-2.2.0-foss-{generation}.eb"
    provider_dir = (
        "packaging/easybuild-overlay"
        if generation == "2024a"
        else f"packaging/easybuild-{generation}-overlay"
    )
    registry = repository / provider_dir / "providers.json"
    return [
        *(f"{recipe}: {error}" for error in validate_root_recipe(recipe, generation)),
        *(
            f"{registry}: {error}"
            for error in validate_provider_registry(
                registry, generation, repository, recipe
            )
        ),
    ]


def validate_robot_receipt(repository: Path) -> list[str]:
    """Validate the retained robot proof without promoting it to build evidence."""
    receipt_path = repository / "packaging/easybuild/root-graph-resolution.json"
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("catalogue_commit") != "58e8b5a48767cbed1bf5669675d9638580d7259f":
        errors.append("robot receipt must bind the pinned easyconfigs catalogue")
    for generation in ("2023a", "2024a"):
        record = data.get("generations", {}).get(generation, {})
        gcccore = "12.3.0" if generation == "2023a" else "13.3.0"
        for artifact in ("root_recipe", "provider_registry", "foundation_manifest"):
            value = record.get(artifact)
            artifact_path = (
                _repository_file(repository, value) if isinstance(value, str) else None
            )
            expected_sha = record.get(f"{artifact}_sha256")
            if artifact_path is None:
                errors.append(f"{generation} {artifact} is missing")
            elif hashlib.sha256(artifact_path.read_bytes()).hexdigest() != expected_sha:
                errors.append(f"{generation} {artifact} SHA-256 does not match")
        log_value = record.get("robot_log")
        log_path = (
            _repository_file(repository, log_value)
            if isinstance(log_value, str)
            else None
        )
        if log_path is None:
            errors.append(f"{generation} robot log is missing")
            continue
        if hashlib.sha256(log_path.read_bytes()).hexdigest() != record.get(
            "robot_log_sha256"
        ):
            errors.append(f"{generation} robot log SHA-256 does not match")
        text = log_path.read_text(encoding="utf-8")
        modules = _robot_modules(text)
        if len(modules) != EXPECTED_ROBOT_MODULE_COUNT:
            errors.append(f"{generation} robot log must resolve 108 modules")
        if record.get("resolved_module_count") != len(modules):
            errors.append(f"{generation} receipt module count does not match its log")
        errors.extend(validate_robot_modules(record, generation, modules))
        required_modules = (
            f"voiage/2.2.0-foss-{generation}",
            f"Python/3.12.14-GCCcore-{gcccore}",
            f"Rust/1.96.0-GCCcore-{gcccore}",
            f"maturin/1.13.1-GCCcore-{gcccore}",
            f"Arrow/25.0.1-gfbf-{generation}",
            f"polars/1.42.1-GCCcore-{gcccore}",
            f"pydantic/2.13.4-GCCcore-{gcccore}",
            f"jsonschema/4.26.0-GCCcore-{gcccore}",
        )
        errors.extend(
            f"{generation} robot log is missing {module}"
            for module in required_modules
            if f"(module: {module})" not in text
        )
        if any(marker in text for marker in ("/Users/", "/Volumes/", "/var/folders/")):
            errors.append(f"{generation} robot log contains a private absolute path")
        if record.get("dependency_resolution_passed") is not True:
            errors.append(f"{generation} dependency resolution must be recorded passed")
        false_fields = (
            "native_build_executed",
            "installed_module_smoke_executed",
            "full_native_voiage_ready",
        )
        errors.extend(
            f"{generation} {field} must remain false"
            for field in false_fields
            if record.get(field) is not False
        )

    generation_2023 = data.get("generations", {}).get("2023a", {})
    if generation_2023.get("python_modules") != ["Python/3.12.14-GCCcore-12.3.0"]:
        errors.append("2023a robot receipt must contain only Python 3.12.14")
    generation_2024 = data.get("generations", {}).get("2024a", {})
    if generation_2024.get("runtime_python_module") != "Python/3.12.14-GCCcore-13.3.0":
        errors.append("2024a runtime Python must remain 3.12.14")
    if (
        generation_2024.get("build_tool_exception")
        != EXPECTED_2024A_BUILD_TOOL_EXCEPTION
    ):
        errors.append(
            "2024a Python 3.12.3 exception must remain narrowly build-tool bound"
        )
    return errors


def main() -> int:
    """Validate both maintained EasyBuild generations."""
    repository = Path(__file__).resolve().parents[1]
    errors = [
        error
        for generation in ("2023a", "2024a")
        for error in validate_generation(repository, generation)
    ]
    errors.extend(validate_robot_receipt(repository))
    if errors:
        print("EasyBuild root graph contract failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("EasyBuild root graph contract passed for 2023a and 2024a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
