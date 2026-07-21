"""Verify the v1 optional and experimental packaging boundary."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_extension_packaging_manifest_is_complete_and_matches_project_extras() -> None:
    manifest = json.loads(
        (ROOT / "specs/v1/extension-packaging.json").read_text(encoding="utf-8")
    )
    project = json.loads("{}")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.11 fallback
        import tomli as tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert manifest["schema_version"] == 1
    assert set(manifest["optional_extras"]) <= set(extras)
    assert set(manifest["stable_base"]["forbidden_extras"]) == {
        "ecosystem",
        "experimental",
        "jax",
        "plotting",
        "performance",
        "deep_learning",
        "web",
    }
    assert manifest["experimental"]["stable_guarantee"] is False


def test_extension_packaging_manifest_uses_only_existing_runtime_roots() -> None:
    manifest = json.loads(
        (ROOT / "specs/v1/extension-packaging.json").read_text(encoding="utf-8")
    )
    for group in ("optional_extras", "experimental"):
        entries = (
            manifest[group].values()
            if group == "optional_extras"
            else [manifest[group]]
        )
        for entry in entries:
            for root in entry["runtime_roots"]:
                assert (ROOT / root).exists(), root
