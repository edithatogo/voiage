"""Runtime packaging parity and isolated external-producer tooling contracts."""

from __future__ import annotations

from pathlib import Path
import shutil
import tomllib

import pytest

ROOT = Path(__file__).parents[1]


def test_published_runtime_has_no_legacy_tabular_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any(d.startswith(("pandas", "xarray")) for d in project["dependencies"])


def test_runtime_packaging_surfaces_agree() -> None:
    from scripts.check_packaging_dependencies import validate

    assert validate(ROOT)["passed"]


@pytest.mark.parametrize(
    "surface",
    [
        "pyproject.toml",
        "conda-recipe/meta.yaml",
        "tox.ini",
        ".github/workflows/dependency-preview.yml",
    ],
)
def test_runtime_drift_is_rejected(tmp_path: Path, surface: str) -> None:
    from scripts.check_packaging_dependencies import validate

    paths = [
        "pyproject.toml",
        "conda-recipe/meta.yaml",
        "tox.ini",
        ".github/workflows/dependency-preview.yml",
        "specs/dependency-runtime-policy.json",
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    path = tmp_path / surface
    text = (
        path.read_text()
        .replace("polars>=2.1.0,<3", "polars>=1,<3")
        .replace("polars >=2.1.0,<3", "polars >=1,<3")
        .replace("polars==2.1.0", "polars==1.0")
    )
    assert text != path.read_text()
    path.write_text(text)
    assert not validate(tmp_path)["passed"]


def test_producer_group_is_unpublished_and_not_a_default_install() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["dependency-groups"]["producer-tests"] == ["pandas>=3.1.0,<4"]
    assert config["tool"]["uv"]["default-groups"] == []
    assert all(
        not d.startswith(("pandas", "xarray"))
        for ds in config["project"]["optional-dependencies"].values()
        for d in ds
    )


def test_test_lanes_select_unpublished_producer_tooling_explicitly() -> None:
    import configparser

    tox = configparser.ConfigParser(interpolation=None)
    tox.read(ROOT / "tox.ini")
    assert tox["testenv"]["dependency_groups"] == "producer-tests"
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        for line in path.read_text().splitlines():
            if "uv sync " in line and "--extra ci" in line:
                assert "--group producer-tests" in line, (path.name, line)
