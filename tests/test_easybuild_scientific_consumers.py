from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from packaging.utils import canonicalize_name
import pytest

ROOT = Path(__file__).parents[1]
OVERLAYS = {
    "2023a": ROOT / "packaging/easybuild-2023a-overlay",
    "2024a": ROOT / "packaging/easybuild-overlay",
}
EXPECTED = {
    "joblib": (
        "1.5.3",
        "8561a3269e6801106863fd0d6d84bb737be9e7631e33aaed3fb9ce5953688da3",
    ),
    "threadpoolctl": (
        "3.6.0",
        "8ab8b4aa3491d812b623328249fab5302a68d2d71745c8a4c719a2fcaba9f44e",
    ),
    "xarray": (
        "2024.11.0",
        "1ccace44573ddb862e210ad3ec204210654d2c750bec11bbe7d842dfc298591f",
    ),
    "scikit-learn": (
        "1.7.2",
        "20e9e49ecd130598f1ca38a1d85090e1a600147b9c02fa6f15d69cb53d968fda",
    ),
}
BUILD_ONLY = {
    "setuptools": (
        "84.0.0",
        "f4695c21257f0d9b537ec2692c941d02ee143b7cc1276941349a546573b2ef73",
    )
}


def _recipe(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)  # type: ignore[attr-defined]
            except (ValueError, TypeError):
                # Easyconfigs may contain computed assignments outside this contract.
                pass
    return values


@pytest.mark.parametrize("generation", ["2023a", "2024a"])
def test_scientific_consumer_recipe_is_exact_and_fail_closed(generation: str) -> None:
    root = OVERLAYS[generation]
    compiler = "12.3.0" if generation == "2023a" else "13.3.0"
    recipe = _recipe(
        root / generation / f"Voiage-scientific-consumers-2.2.0-gfbf-{generation}.eb"
    )
    assert recipe["toolchain"] == {"name": "gfbf", "version": generation}
    assert recipe["dependencies"] == [
        ("Python", "3.12.14"),
        ("SciPy-bundle", "2026.09", "-voiage-2.2.0"),
        ("Voiage-Python-support", "2.2.0"),
    ]
    assert recipe["builddependencies"] == [
        ("Voiage-scientific-build-support", "2.2.0"),
        ("Cython", "3.0.10"),
        ("meson-python", "0.16.0"),
    ]
    assert all(
        recipe[key] is True
        for key in [
            "use_pip",
            "pip_no_index",
            "pip_no_build_isolation",
            "pip_ignore_installed",
            "download_dep_fail",
            "sanity_pip_check",
        ]
    )
    extensions = {
        canonicalize_name(name): (version, options)
        for name, version, options in recipe["exts_list"]
    }
    assert list(extensions) == list(EXPECTED)
    for name, (version, digest) in EXPECTED.items():
        assert extensions[name][0] == version
        assert extensions[name][1]["checksums"] == [digest]
    assert "LinearRegression" in recipe["sanity_check_commands"][0]
    assert (
        compiler in next((root / generation).glob("Python-3.12.14-GCCcore-*.eb")).name
    )


@pytest.mark.parametrize("generation", ["2023a", "2024a"])
def test_source_and_robot_evidence_is_bound_without_native_claim(
    generation: str,
) -> None:
    root = OVERLAYS[generation]
    evidence = json.loads((root / "scientific-consumer-sources.json").read_text())
    assert evidence["toolchain_generation"] == generation
    assert evidence["native_builds_executed"] is False
    assert {
        canonicalize_name(x["name"]): (x["version"], x["sha256"])
        for x in evidence["sources"]
    } == BUILD_ONLY | EXPECTED
    assert all(
        x["download_hash_verified"] and x["bytes"] > 0 for x in evidence["sources"]
    )
    sklearn = next(x for x in evidence["sources"] if x["name"] == "scikit-learn")
    assert sklearn["build_system"]["build-backend"] == "mesonpy"
    assert {"joblib>=1.2.0", "threadpoolctl>=3.1.0"}.issubset(sklearn["requires_dist"])
    log = (root / "evidence/scientific-consumers-robot.log").read_text()
    assert "Voiage-scientific-consumers/2.2.0-gfbf-" + generation in log
    assert "Dry run: printing build status" in log
    assert "/Volumes/" not in log
    assert "/Users/" not in log
    assert "/var/folders/" not in log
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        manifest["files"]["scientific-consumer-sources.json"]
        == hashlib.sha256(
            (root / "scientific-consumer-sources.json").read_bytes()
        ).hexdigest()
    )
    assert (
        manifest["files"]["evidence/scientific-consumers-robot.log"]
        == hashlib.sha256(
            (root / "evidence/scientific-consumers-robot.log").read_bytes()
        ).hexdigest()
    )
    assert not manifest.get(
        "native_scientific_build",
        manifest.get("native_python_or_scientific_builds_executed"),
    )
    assert not manifest.get("full_voiage_graph", manifest.get("full_voiage_ready"))
