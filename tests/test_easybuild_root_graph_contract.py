"""Red-phase contracts for final EasyBuild root graph reconciliation."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_easybuild_root_graph import (
    EXPECTED_BUILD_DEPENDENCIES,
    EXPECTED_RUNTIME_DEPENDENCIES,
    validate_generation,
    validate_provider_registry,
    validate_robot_modules,
    validate_robot_receipt,
    validate_root_recipe,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("generation", ["2023a", "2024a"])
def test_current_root_graph_has_no_stale_dependency_or_provider_contracts(
    generation: str,
) -> None:
    """Remain red until the generation's root graph is fully reconciled."""
    assert validate_generation(ROOT, generation) == []


def test_robot_receipt_binds_resolved_graph_and_build_tool_exception() -> None:
    assert validate_robot_receipt(ROOT) == []


@pytest.mark.parametrize(
    ("generation", "modules", "expected"),
    [
        (
            "2023a",
            ["Python/3.12.14-GCCcore-12.3.0", "Python/3.11.0-GCCcore-12.3.0"],
            "receipt Python modules do not match its log",
        ),
        (
            "2023a",
            ["Python/3.12.14-GCCcore-12.3.0"] * 2,
            "robot log contains duplicate modules",
        ),
        (
            "2024a",
            [
                "Python/3.12.14-GCCcore-13.3.0",
                "Python/3.12.3-GCCcore-13.3.0",
                "libpciaccess/0.18.1-GCCcore-13.3.0",
            ],
            "build-tool exception consumer is missing from its robot log",
        ),
    ],
)
def test_robot_module_validator_rejects_unbound_receipt_claims(
    generation: str, modules: list[str], expected: str
) -> None:
    receipt = json.loads(
        (ROOT / "packaging/easybuild/root-graph-resolution.json").read_text(
            encoding="utf-8"
        )
    )
    record = receipt["generations"][generation]
    assert expected in "\n".join(validate_robot_modules(record, generation, modules))


def test_2024a_legacy_python_cannot_enter_root_runtime_dependencies() -> None:
    receipt = json.loads(
        (ROOT / "packaging/easybuild/root-graph-resolution.json").read_text(
            encoding="utf-8"
        )
    )
    record = receipt["generations"]["2024a"]
    assert record["runtime_python_module"] == "Python/3.12.14-GCCcore-13.3.0"
    assert record["build_tool_exception"] == {
        "module": "Python/3.12.3-GCCcore-13.3.0",
        "consumer": "Meson/1.4.0-GCCcore-13.3.0",
        "consumer_role": "builddependency",
        "transitive_consumer": "libpciaccess/0.18.1-GCCcore-13.3.0",
        "transitive_consumer_role": "builddependency",
    }
    root = _assignments_for_test(
        ROOT / "packaging/easybuild/voiage-2.2.0-foss-2024a.eb"
    )
    assert ("Python", "3.12.3") not in root["dependencies"]


def _assignments_for_test(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (TypeError, ValueError):
                        # Non-literal assignments are irrelevant to this test helper.
                        pass
    return values


def _write_recipe(path: Path, build: object, runtime: object) -> None:
    path.write_text(
        "toolchain = {'name': 'foss', 'version': '2024a'}\n"
        f"builddependencies = {build!r}\n"
        f"dependencies = {runtime!r}\n",
        encoding="utf-8",
    )
    ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("section", "replacement", "expected"),
    [
        ("runtime", ("Python", "3.12.3"), "Python', '3.12.3"),
        ("build-rust", ("Rust", "1.85.0"), "Rust', '1.85.0"),
        ("build-maturin", ("maturin", "1.9.6"), "maturin', '1.9.6"),
        ("runtime", ("NumPy", "2.2.6"), "stale direct modules"),
    ],
)
def test_validator_rejects_stale_root_modules(
    tmp_path: Path, section: str, replacement: tuple[str, str], expected: str
) -> None:
    build = list(EXPECTED_BUILD_DEPENDENCIES)
    runtime = [
        tuple(item) if item[2] else tuple(item[:2])
        for item in EXPECTED_RUNTIME_DEPENDENCIES
    ]
    if section == "runtime":
        runtime[0 if replacement[0] == "Python" else 1] = replacement
    elif section == "build-rust":
        build[0] = replacement
    else:
        build[1] = replacement
    recipe = tmp_path / "voiage-2.2.0-foss-2024a.eb"
    _write_recipe(recipe, build, runtime)
    assert expected in "\n".join(validate_root_recipe(recipe, "2024a"))


def test_validator_accepts_generation_neutral_current_contract(tmp_path: Path) -> None:
    recipe = tmp_path / "voiage-2.2.0-foss-2024a.eb"
    runtime = [
        tuple(item) if item[2] else tuple(item[:2])
        for item in EXPECTED_RUNTIME_DEPENDENCIES
    ]
    _write_recipe(recipe, list(EXPECTED_BUILD_DEPENDENCIES), runtime)
    assert validate_root_recipe(recipe, "2024a") == []


def _write_provider_contract(repository: Path, recipe: Path) -> Path:
    manifest = repository / "packaging/easybuild-2024a-rust-overlay/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    recipe_sha = hashlib.sha256(recipe.read_bytes()).hexdigest()
    registry = repository / "providers.json"
    providers = {
        "pydantic": ("rust", "2.13.4", "pydantic/2.13.4"),
        "jsonschema": ("rust", "4.26.0", "jsonschema/4.26.0"),
        "polars": ("polars", "1.42.1", "polars/1.42.1"),
        "pyarrow": ("arrow", "25.0.1", "Arrow/25.0.1"),
    }
    data = {
        "toolchain_generation": "2024a",
        "external_provider_overlays": {
            name: f"packaging/easybuild-2024a-{fragment}-overlay"
            for name, (fragment, _, _) in providers.items()
        },
        "external_provider_evidence": {
            name: {
                "version": version,
                "manifest": manifest.relative_to(repository).as_posix(),
                "manifest_sha256": manifest_sha,
                "native_build_executed": False,
                "full_voiage_ready": False,
                "consumer_recipe": recipe.relative_to(repository).as_posix(),
                "consumer_recipe_sha256": recipe_sha,
                "consumer_dependency": dependency,
            }
            for name, (_, version, dependency) in providers.items()
        },
    }
    registry.write_text(json.dumps(data), encoding="utf-8")
    return registry


def test_provider_validator_accepts_exact_generation_neutral_bindings(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "packaging/easybuild/voiage-2.2.0-foss-2024a.eb"
    recipe.parent.mkdir(parents=True)
    _write_recipe(recipe, [], [])
    registry = _write_provider_contract(tmp_path, recipe)
    assert validate_provider_registry(registry, "2024a", tmp_path, recipe) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("manifest_sha256", "0" * 64, "manifest SHA-256 does not match"),
        ("consumer_recipe_sha256", "0" * 64, "current root recipe SHA-256"),
        ("consumer_dependency", "PyArrow/25.0.1", "consumer dependency"),
        ("native_build_executed", True, "native-build status must remain false"),
        ("full_voiage_ready", True, "full-graph status must remain false"),
    ],
)
def test_provider_validator_rejects_mutated_evidence(
    tmp_path: Path, field: str, value: object, expected: str
) -> None:
    recipe = tmp_path / "packaging/easybuild/voiage-2.2.0-foss-2024a.eb"
    recipe.parent.mkdir(parents=True)
    _write_recipe(recipe, [], [])
    registry = _write_provider_contract(tmp_path, recipe)
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["external_provider_evidence"]["pyarrow"][field] = value
    registry.write_text(json.dumps(data), encoding="utf-8")
    assert expected in "\n".join(
        validate_provider_registry(registry, "2024a", tmp_path, recipe)
    )


def test_provider_validator_rejects_manifest_path_escape(tmp_path: Path) -> None:
    recipe = tmp_path / "packaging/easybuild/voiage-2.2.0-foss-2024a.eb"
    recipe.parent.mkdir(parents=True)
    _write_recipe(recipe, [], [])
    registry = _write_provider_contract(tmp_path, recipe)
    outside = tmp_path.parent / "outside-manifest.json"
    outside.write_text("{}\n", encoding="utf-8")
    data = json.loads(registry.read_text(encoding="utf-8"))
    record = data["external_provider_evidence"]["pyarrow"]
    record["manifest"] = "../outside-manifest.json"
    record["manifest_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    registry.write_text(json.dumps(data), encoding="utf-8")
    assert "evidence manifest is missing" in "\n".join(
        validate_provider_registry(registry, "2024a", tmp_path, recipe)
    )
