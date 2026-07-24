#!/usr/bin/env python3
"""Compose and validate voiage's deterministic mixed-language CycloneDX SBOM."""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

CYCLONEDX_SCHEMA = "http://cyclonedx.org/schema/bom-1.6.schema.json"
SPEC_VERSION = "1.6"
SCOPE_PROPERTIES = {
    "voiage:sbom:scope": "mixed-language-release",
    "voiage:sbom:ecosystems": "python,cargo,r,julia",
    "voiage:sbom:python": "resolved-installed-runtime-environment",
    "voiage:sbom:cargo": "resolved-cargo-lock-workspace",
    "voiage:sbom:r": "declared-description-dependencies",
    "voiage:sbom:julia": "declared-project-dependencies",
}
REQUIRED_ECOSYSTEMS = ("python", "cargo", "r", "julia")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
R_DEPENDENCY = re.compile(r"^([A-Za-z][A-Za-z0-9.]*)\s*(?:\(([^)]+)\))?$")

JsonObject = dict[str, Any]


class SbomError(ValueError):
    """Raised when an input or composed SBOM violates the composition contract."""


def _read_json(path: Path) -> JsonObject:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SbomError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SbomError(f"{path} must contain a JSON object")
    return document


def _read_toml(path: Path) -> JsonObject:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SbomError(f"cannot read TOML from {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SbomError(f"{path} must contain a TOML table")
    return document


def _property_map(properties: object) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(properties, list):
        return result
    for item in properties:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        if isinstance(name, str) and isinstance(value, str):
            result[name] = value
    return result


def _set_properties(target: JsonObject, values: Mapping[str, str]) -> None:
    properties = _property_map(target.get("properties"))
    properties.update(values)
    target["properties"] = [
        {"name": name, "value": value} for name, value in sorted(properties.items())
    ]


def _component_ref(component: Mapping[str, Any], *, context: str) -> str:
    reference = component.get("bom-ref")
    if not isinstance(reference, str) or not reference:
        raise SbomError(f"{context} component is missing a non-empty bom-ref")
    return reference


def _cargo_ref(name: str, version: str) -> str:
    return f"pkg:cargo/{quote(name, safe='')}@{quote(version, safe='')}"


def _r_ref(name: str, version: str | None = None) -> str:
    base = f"pkg:cran/{quote(name, safe='')}"
    return f"{base}@{quote(version, safe='')}" if version else base


def _julia_ref(
    name: str,
    *,
    uuid: str,
    version: str | None = None,
) -> str:
    base = f"pkg:julia/{quote(name, safe='')}"
    if version:
        base = f"{base}@{quote(version, safe='')}"
    return f"{base}?uuid={quote(uuid, safe='')}"


def _dependency_entry(reference: str, depends_on: Iterable[str] = ()) -> JsonObject:
    dependencies = sorted(set(depends_on))
    entry: JsonObject = {"ref": reference}
    if dependencies:
        entry["dependsOn"] = dependencies
    return entry


def _parse_cargo_dependency(
    dependency: str,
    packages_by_name: Mapping[str, Sequence[JsonObject]],
) -> JsonObject:
    parts = dependency.split()
    if not parts:
        raise SbomError("Cargo.lock contains an empty dependency")
    name = parts[0]
    version = parts[1] if len(parts) >= 2 else None
    candidates = list(packages_by_name.get(name, ()))
    if version is not None:
        candidates = [
            package for package in candidates if package.get("version") == version
        ]
    if len(candidates) != 1:
        qualifier = f" {version}" if version else ""
        raise SbomError(
            f"Cargo.lock dependency {name}{qualifier} resolves to "
            f"{len(candidates)} packages"
        )
    return candidates[0]


def _workspace_metadata(cargo_workspace: Path) -> dict[str, JsonObject]:
    root_manifest = _read_toml(cargo_workspace / "Cargo.toml")
    workspace = root_manifest.get("workspace")
    if not isinstance(workspace, dict):
        raise SbomError("rust/Cargo.toml is missing [workspace]")
    package_defaults = workspace.get("package", {})
    if not isinstance(package_defaults, dict):
        package_defaults = {}

    members = workspace.get("members")
    if not isinstance(members, list) or not all(
        isinstance(member, str) for member in members
    ):
        raise SbomError("rust/Cargo.toml workspace.members must be a string list")

    result: dict[str, JsonObject] = {}
    for member in members:
        manifest_path = cargo_workspace / member / "Cargo.toml"
        manifest = _read_toml(manifest_path)
        package = manifest.get("package")
        if not isinstance(package, dict) or not isinstance(package.get("name"), str):
            raise SbomError(f"{manifest_path} is missing [package].name")
        metadata = dict(package)
        for field in ("version", "license", "repository"):
            value = metadata.get(field)
            if isinstance(value, dict) and value.get("workspace") is True:
                metadata[field] = package_defaults.get(field)
        result[metadata["name"]] = metadata
    return result


def _cargo_inventory(
    cargo_workspace: Path,
) -> tuple[list[JsonObject], list[JsonObject], list[str]]:
    lock_path = cargo_workspace / "Cargo.lock"
    lock = _read_toml(lock_path)
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise SbomError(f"{lock_path} has no [[package]] entries")

    package_objects: list[JsonObject] = []
    packages_by_name: dict[str, list[JsonObject]] = {}
    for raw in packages:
        if not isinstance(raw, dict):
            raise SbomError(f"{lock_path} contains a non-table package entry")
        name = raw.get("name")
        version = raw.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise SbomError(f"{lock_path} package entries require name and version")
        package = dict(raw)
        package_objects.append(package)
        packages_by_name.setdefault(name, []).append(package)

    workspace_metadata = _workspace_metadata(cargo_workspace)
    components: list[JsonObject] = []
    dependencies: list[JsonObject] = []
    workspace_refs: list[str] = []

    for package in package_objects:
        name = package["name"]
        version = package["version"]
        reference = _cargo_ref(name, version)
        component: JsonObject = {
            "bom-ref": reference,
            "name": name,
            "purl": reference,
            "type": "library",
            "version": version,
        }
        properties = {
            "voiage:ecosystem": "cargo",
            "voiage:inventory:resolution": "resolved-lock",
            "voiage:inventory:source": "rust/Cargo.lock",
        }

        checksum = package.get("checksum")
        if isinstance(checksum, str):
            if not HEX_64.fullmatch(checksum):
                raise SbomError(f"invalid Cargo SHA-256 checksum for {name} {version}")
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]

        source = package.get("source")
        if isinstance(source, str):
            properties["voiage:cargo:source"] = source

        local_metadata = workspace_metadata.get(name)
        if local_metadata is not None:
            properties["voiage:cargo:workspace-member"] = "true"
            workspace_refs.append(reference)
            description = local_metadata.get("description")
            if isinstance(description, str):
                component["description"] = description
            license_id = local_metadata.get("license")
            if isinstance(license_id, str):
                component["licenses"] = [{"license": {"id": license_id}}]
            repository = local_metadata.get("repository")
            if isinstance(repository, str):
                component["externalReferences"] = [{"type": "vcs", "url": repository}]

        _set_properties(component, properties)
        components.append(component)

        raw_dependencies = package.get("dependencies", [])
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) for item in raw_dependencies
        ):
            raise SbomError(f"Cargo dependencies for {name} must be strings")
        dependency_refs = [
            _cargo_ref(candidate["name"], candidate["version"])
            for item in raw_dependencies
            for candidate in [_parse_cargo_dependency(item, packages_by_name)]
        ]
        dependencies.append(_dependency_entry(reference, dependency_refs))

    return components, dependencies, sorted(workspace_refs)


def _parse_r_dependencies(value: str) -> list[tuple[str, str | None]]:
    dependencies: list[tuple[str, str | None]] = []
    for raw in value.replace("\n", " ").split(","):
        item = raw.strip()
        if not item:
            continue
        match = R_DEPENDENCY.fullmatch(item)
        if match is None:
            raise SbomError(f"cannot parse R dependency declaration: {item}")
        name, constraint = match.groups()
        if name == "R":
            continue
        dependencies.append((name, constraint.strip() if constraint else None))
    return dependencies


def _r_inventory(
    description_path: Path,
    *,
    rust_ffi_ref: str | None,
) -> tuple[list[JsonObject], list[JsonObject], str]:
    try:
        fields = Parser().parsestr(description_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SbomError(f"cannot read {description_path}: {exc}") from exc
    name = fields.get("Package")
    version = fields.get("Version")
    if not name or not version:
        raise SbomError(f"{description_path} requires Package and Version")

    root_ref = _r_ref(name, version)
    root: JsonObject = {
        "bom-ref": root_ref,
        "name": name,
        "purl": root_ref,
        "type": "library",
        "version": version,
    }
    description = fields.get("Description")
    if description:
        root["description"] = " ".join(description.split())
    license_value = fields.get("License")
    if license_value:
        root["licenses"] = [{"license": {"name": license_value.strip()}}]
    _set_properties(
        root,
        {
            "voiage:ecosystem": "r",
            "voiage:inventory:resolution": "declared",
            "voiage:inventory:source": "r-package/voiageR/DESCRIPTION",
        },
    )

    components = [root]
    direct_refs: list[str] = []
    dependency_fields = {
        "Depends": "required",
        "Imports": "required",
        "LinkingTo": "required",
        "Suggests": "optional",
        "Enhances": "optional",
    }
    seen: dict[str, JsonObject] = {}
    for field, scope in dependency_fields.items():
        value = fields.get(field)
        if not value:
            continue
        for dependency_name, constraint in _parse_r_dependencies(value):
            reference = _r_ref(dependency_name)
            direct_refs.append(reference)
            component = seen.setdefault(
                reference,
                {
                    "bom-ref": reference,
                    "name": dependency_name,
                    "purl": reference,
                    "type": "library",
                },
            )
            if scope == "required" or component.get("scope") != "required":
                component["scope"] = scope
            properties = {
                "voiage:ecosystem": "r",
                "voiage:inventory:dependency-field": field,
                "voiage:inventory:resolution": "declared-unresolved",
                "voiage:inventory:source": "r-package/voiageR/DESCRIPTION",
            }
            if constraint:
                properties["voiage:inventory:version-constraint"] = constraint
            _set_properties(component, properties)

    components.extend(seen.values())
    if rust_ffi_ref:
        direct_refs.append(rust_ffi_ref)
        _set_properties(root, {"voiage:binding:native-component": rust_ffi_ref})
    dependencies = [_dependency_entry(root_ref, direct_refs)]
    dependencies.extend(_dependency_entry(reference) for reference in sorted(seen))
    return components, dependencies, root_ref


def _julia_inventory(
    project_path: Path,
    *,
    rust_ffi_ref: str | None,
) -> tuple[list[JsonObject], list[JsonObject], str]:
    project = _read_toml(project_path)
    name = project.get("name")
    uuid = project.get("uuid")
    version = project.get("version")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(uuid, str)
        or not uuid
        or not isinstance(version, str)
        or not version
    ):
        raise SbomError(f"{project_path} requires name, uuid, and version")

    root_ref = _julia_ref(name, uuid=uuid, version=version)
    root: JsonObject = {
        "bom-ref": root_ref,
        "name": name,
        "purl": root_ref,
        "type": "library",
        "version": version,
    }
    _set_properties(
        root,
        {
            "voiage:ecosystem": "julia",
            "voiage:inventory:resolution": "declared",
            "voiage:inventory:source": "bindings/julia/Project.toml",
            "voiage:julia:uuid": uuid,
        },
    )

    compat = project.get("compat", {})
    if not isinstance(compat, dict):
        raise SbomError(f"{project_path} [compat] must be a table")
    components = [root]
    direct_refs: list[str] = []
    seen: set[str] = set()
    for table_name, scope in (("deps", "required"), ("extras", "optional")):
        table = project.get(table_name, {})
        if not isinstance(table, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in table.items()
        ):
            raise SbomError(f"{project_path} [{table_name}] must map names to UUIDs")
        typed_table = {
            key: value
            for key, value in table.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        for dependency_name, dependency_uuid in sorted(typed_table.items()):
            reference = _julia_ref(dependency_name, uuid=dependency_uuid)
            direct_refs.append(reference)
            if reference in seen:
                continue
            seen.add(reference)
            component: JsonObject = {
                "bom-ref": reference,
                "name": dependency_name,
                "purl": reference,
                "scope": scope,
                "type": "library",
            }
            properties: dict[str, str] = {
                "voiage:ecosystem": "julia",
                "voiage:inventory:dependency-field": table_name,
                "voiage:inventory:resolution": "declared-unresolved",
                "voiage:inventory:source": "bindings/julia/Project.toml",
                "voiage:julia:uuid": dependency_uuid,
            }
            constraint = compat.get(dependency_name)
            if isinstance(constraint, str):
                properties["voiage:inventory:version-constraint"] = constraint
            _set_properties(component, properties)
            components.append(component)

    julia_constraint = compat.get("julia")
    if isinstance(julia_constraint, str):
        _set_properties(root, {"voiage:julia:version-constraint": julia_constraint})
    if rust_ffi_ref:
        direct_refs.append(rust_ffi_ref)
        _set_properties(root, {"voiage:binding:native-component": rust_ffi_ref})
    dependencies = [_dependency_entry(root_ref, direct_refs)]
    dependencies.extend(_dependency_entry(reference) for reference in sorted(seen))
    return components, dependencies, root_ref


def _merge_dependencies(
    dependency_groups: Iterable[Iterable[JsonObject]],
) -> list[JsonObject]:
    graph: dict[str, set[str]] = {}
    for group in dependency_groups:
        for entry in group:
            reference = entry.get("ref")
            if not isinstance(reference, str) or not reference:
                raise SbomError("dependency entries require a non-empty ref")
            depends_on = entry.get("dependsOn", [])
            if not isinstance(depends_on, list) or not all(
                isinstance(item, str) and item for item in depends_on
            ):
                raise SbomError(f"dependency {reference} has invalid dependsOn")
            graph.setdefault(reference, set()).update(depends_on)
    return [
        _dependency_entry(reference, graph[reference]) for reference in sorted(graph)
    ]


def compose_sbom(
    *,
    python_sbom: Path,
    cargo_workspace: Path,
    r_description: Path,
    julia_project: Path,
    source_version: str,
    source_commit: str,
    source_tag: str,
) -> JsonObject:
    """Compose one deterministic CycloneDX 1.6 document from all bindings."""
    document = _read_json(python_sbom)
    if document.get("bomFormat") != "CycloneDX":
        raise SbomError("Python input is not a CycloneDX BOM")
    if document.get("specVersion") != SPEC_VERSION:
        raise SbomError(f"Python input must use CycloneDX {SPEC_VERSION}")

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise SbomError("Python input is missing metadata")
    root = metadata.get("component")
    if not isinstance(root, dict):
        raise SbomError("Python input is missing metadata.component")
    root_ref = _component_ref(root, context="metadata")
    root["version"] = source_version
    _set_properties(
        root,
        {
            "voiage:ecosystem": "python",
            "voiage:inventory:resolution": "resolved-installed",
            "voiage:sbom:aggregate-root": "true",
        },
    )
    _set_properties(
        metadata,
        {
            **SCOPE_PROPERTIES,
            "voiage:source:commit": source_commit,
            "voiage:source:tag": source_tag,
        },
    )

    python_components = document.get("components", [])
    python_dependencies = document.get("dependencies", [])
    if not isinstance(python_components, list) or not isinstance(
        python_dependencies, list
    ):
        raise SbomError("Python input components and dependencies must be arrays")
    for component in python_components:
        if not isinstance(component, dict):
            raise SbomError("Python input contains a non-object component")
        _set_properties(
            component,
            {
                "voiage:ecosystem": "python",
                "voiage:inventory:resolution": "resolved-installed",
            },
        )

    cargo_components, cargo_dependencies, workspace_refs = _cargo_inventory(
        cargo_workspace
    )
    rust_ffi_refs = [
        reference
        for reference in workspace_refs
        if reference.startswith("pkg:cargo/voiage-ffi@")
    ]
    if len(rust_ffi_refs) != 1:
        raise SbomError("Cargo workspace must contain exactly one voiage-ffi package")
    rust_ffi_ref = rust_ffi_refs[0]
    r_components, r_dependencies, r_root = _r_inventory(
        r_description,
        rust_ffi_ref=rust_ffi_ref,
    )
    julia_components, julia_dependencies, julia_root = _julia_inventory(
        julia_project,
        rust_ffi_ref=rust_ffi_ref,
    )
    if source_tag:
        if source_tag != f"v{source_version}":
            raise SbomError(
                f"release tag {source_tag} does not match version {source_version}"
            )
        first_party_components = [
            component
            for component in cargo_components
            if _property_map(component.get("properties")).get(
                "voiage:cargo:workspace-member"
            )
            == "true"
        ]
        first_party_components.extend((r_components[0], julia_components[0]))
        mismatches = [
            f"{component['name']}={component.get('version', '<missing>')}"
            for component in first_party_components
            if component.get("version") != source_version
        ]
        if mismatches:
            raise SbomError(
                "release binding versions do not match source version "
                f"{source_version}: {', '.join(sorted(mismatches))}"
            )

    all_components = [
        *python_components,
        *cargo_components,
        *r_components,
        *julia_components,
    ]
    component_by_ref: dict[str, JsonObject] = {}
    for component in all_components:
        if not isinstance(component, dict):
            raise SbomError("component inventory contains a non-object")
        reference = _component_ref(component, context="inventory")
        if reference in component_by_ref:
            raise SbomError(f"duplicate component bom-ref: {reference}")
        component_by_ref[reference] = component

    root_direct = [*workspace_refs, r_root, julia_root]
    root_dependency = _dependency_entry(root_ref, root_direct)
    document["components"] = [
        component_by_ref[reference] for reference in sorted(component_by_ref)
    ]
    document["dependencies"] = _merge_dependencies(
        (
            python_dependencies,
            cargo_dependencies,
            r_dependencies,
            julia_dependencies,
            [root_dependency],
        )
    )
    document["$schema"] = CYCLONEDX_SCHEMA
    document["bomFormat"] = "CycloneDX"
    document["specVersion"] = SPEC_VERSION
    document["version"] = 1
    document.pop("serialNumber", None)
    validate_sbom(
        document,
        expected_commit=source_commit,
        expected_tag=source_tag,
        expected_version=source_version,
    )
    return document


def validate_sbom(
    document: JsonObject,
    *,
    expected_commit: str | None = None,
    expected_tag: str | None = None,
    expected_version: str | None = None,
) -> None:
    """Validate mixed-language coverage and graph invariants."""
    errors: list[str] = []
    if document.get("$schema") != CYCLONEDX_SCHEMA:
        errors.append(f"$schema must be {CYCLONEDX_SCHEMA}")
    if document.get("bomFormat") != "CycloneDX":
        errors.append("bomFormat must be CycloneDX")
    if document.get("specVersion") != SPEC_VERSION:
        errors.append(f"specVersion must be {SPEC_VERSION}")
    if document.get("version") != 1:
        errors.append("BOM version must be 1")

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}
    root = metadata.get("component")
    if not isinstance(root, dict):
        errors.append("metadata.component must be an object")
        root = {}
    try:
        root_ref = _component_ref(root, context="metadata")
    except SbomError as exc:
        errors.append(str(exc))
        root_ref = ""

    metadata_properties = _property_map(metadata.get("properties"))
    for name, value in SCOPE_PROPERTIES.items():
        if metadata_properties.get(name) != value:
            errors.append(f"metadata property {name} must equal {value}")
    if (
        expected_commit is not None
        and metadata_properties.get("voiage:source:commit") != expected_commit
    ):
        errors.append("source commit property does not match")
    if (
        expected_tag is not None
        and metadata_properties.get("voiage:source:tag") != expected_tag
    ):
        errors.append("source tag property does not match")
    if expected_version is not None and root.get("version") != expected_version:
        errors.append("metadata component version does not match")

    components = document.get("components")
    if not isinstance(components, list):
        errors.append("components must be an array")
        components = []
    refs: set[str] = set()
    root_ecosystem = _property_map(root.get("properties")).get("voiage:ecosystem")
    ecosystems = {root_ecosystem} if root_ecosystem else set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            errors.append(f"components[{index}] must be an object")
            continue
        try:
            reference = _component_ref(component, context=f"components[{index}]")
        except SbomError as exc:
            errors.append(str(exc))
            continue
        if reference in refs:
            errors.append(f"duplicate component bom-ref: {reference}")
        refs.add(reference)
        ecosystem = _property_map(component.get("properties")).get("voiage:ecosystem")
        if ecosystem:
            ecosystems.add(ecosystem)
        hashes = component.get("hashes", [])
        if not isinstance(hashes, list):
            errors.append(f"component {reference} hashes must be an array")
        else:
            for hash_entry in hashes:
                if not isinstance(hash_entry, dict):
                    errors.append(f"component {reference} has a non-object hash")
                    continue
                if hash_entry.get("alg") == "SHA-256" and not HEX_64.fullmatch(
                    str(hash_entry.get("content", ""))
                ):
                    errors.append(f"component {reference} has invalid SHA-256")

    missing_ecosystems = set(REQUIRED_ECOSYSTEMS) - ecosystems
    if missing_ecosystems:
        errors.append(
            "component inventory is missing ecosystems: "
            + ", ".join(sorted(missing_ecosystems))
        )

    known_refs = {*refs}
    if root_ref:
        known_refs.add(root_ref)
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list):
        errors.append("dependencies must be an array")
        dependencies = []
    graph_refs: set[str] = set()
    for index, entry in enumerate(dependencies):
        if not isinstance(entry, dict):
            errors.append(f"dependencies[{index}] must be an object")
            continue
        reference = entry.get("ref")
        if not isinstance(reference, str) or not reference:
            errors.append(f"dependencies[{index}] requires a non-empty ref")
            continue
        if reference in graph_refs:
            errors.append(f"duplicate dependency graph ref: {reference}")
        graph_refs.add(reference)
        if reference not in known_refs:
            errors.append(f"dependency graph contains unknown ref: {reference}")
        depends_on = entry.get("dependsOn", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(item, str) and item for item in depends_on
        ):
            errors.append(f"dependency {reference} has invalid dependsOn")
            continue
        if depends_on != sorted(set(depends_on)):
            errors.append(f"dependency {reference} dependsOn is not canonical")
        errors.extend(
            (f"dependency {reference} points to unknown ref: {dependency_ref}")
            for dependency_ref in depends_on
            if dependency_ref not in known_refs
        )
    missing_graph_entries = refs - graph_refs
    if missing_graph_entries:
        errors.append(
            "components are missing dependency graph entries: "
            + ", ".join(sorted(missing_graph_entries))
        )

    sorted_refs = [
        component.get("bom-ref")
        for component in components
        if isinstance(component, dict)
    ]
    if sorted_refs != sorted(sorted_refs):
        errors.append("components are not sorted by bom-ref")
    dependency_order = [
        entry.get("ref") for entry in dependencies if isinstance(entry, dict)
    ]
    if dependency_order != sorted(dependency_order):
        errors.append("dependencies are not sorted by ref")

    if errors:
        raise SbomError("\n".join(f"- {error}" for error in errors))


def _write_document(path: Path, document: JsonObject) -> None:
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compose = subparsers.add_parser("compose", help="compose the mixed-language SBOM")
    compose.add_argument("--python-sbom", type=Path, required=True)
    compose.add_argument("--cargo-workspace", type=Path, required=True)
    compose.add_argument("--r-description", type=Path, required=True)
    compose.add_argument("--julia-project", type=Path, required=True)
    compose.add_argument("--source-version", required=True)
    compose.add_argument("--source-commit", required=True)
    compose.add_argument("--source-tag", default="")
    compose.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate scope and graph")
    validate.add_argument("sbom", type=Path)
    validate.add_argument("--expected-version")
    validate.add_argument("--expected-commit")
    validate.add_argument("--expected-tag")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the composition command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "compose":
            document = compose_sbom(
                python_sbom=args.python_sbom,
                cargo_workspace=args.cargo_workspace,
                r_description=args.r_description,
                julia_project=args.julia_project,
                source_version=args.source_version,
                source_commit=args.source_commit,
                source_tag=args.source_tag,
            )
            _write_document(args.output, document)
            print(
                f"wrote {args.output} with {len(document['components'])} components "
                f"(sha256:{_digest(args.output)})"
            )
        else:
            document = _read_json(args.sbom)
            validate_sbom(
                document,
                expected_commit=args.expected_commit,
                expected_tag=args.expected_tag,
                expected_version=args.expected_version,
            )
            print(
                f"validated {args.sbom}: {len(document['components'])} components, "
                f"{len(document['dependencies'])} dependency nodes"
            )
    except SbomError as exc:
        print(f"SBOM error:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
