"""Non-vacuous controls for source bans and installed-package absence."""

from __future__ import annotations

import configparser
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.check_pandas_free import IMPORT_PROBE, scan_source

ROOT = Path(__file__).parents[1]


def test_runtime_source_has_no_legacy_imports() -> None:
    assert scan_source(ROOT / "voiage") == []


@pytest.mark.parametrize(
    "statement",
    [
        "import pandas",
        "import pandas.core as legacy",
        "from pandas import DataFrame",
        "import xarray",
        "from xarray.core import Dataset",
        "lazy import pandas",
        "if False:\n    import xarray",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas",
        "import pandas  # noqa: TID251\n",
        "# ruff: noqa\nimport xarray",
    ],
)
def test_source_lint_rejects_reintroduced_import_without_exemptions(
    tmp_path: Path, statement: str
) -> None:
    nested = tmp_path / "private" / "nested.py"
    nested.parent.mkdir()
    nested.write_text(statement + "\n")
    result = scan_source(tmp_path)
    assert len(result) == 1
    assert result[0]["path"] == str(nested)


def test_source_lint_does_not_mistake_metadata_or_templates_for_imports(
    tmp_path: Path,
) -> None:
    (tmp_path / "metadata.py").write_text(
        'metadata = {"pandas": {}}\ntemplate = "import pandas"\n'
    )
    assert scan_source(tmp_path) == []


def test_source_lint_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No Python sources"):
        scan_source(tmp_path)


def _probe_fixture(
    tmp_path: Path,
    body: str,
    *,
    distribution: str | None = None,
    expected: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    package = tmp_path / "voiage"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.0"\n' + body)
    info = tmp_path / "voiage-0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: voiage\nVersion: 0.0\n")
    for name in ("pandas", "xarray"):
        (tmp_path / f"{name}.py").write_text(
            "# Fictional producer for the negative control.\n"
        )
    if distribution:
        legacy = tmp_path / f"{distribution}-0.0.dist-info"
        legacy.mkdir()
        (legacy / "METADATA").write_text(f"Name: {distribution}\nVersion: 0.0\n")
    code = f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + IMPORT_PROBE
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(expected or tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_probe_accepts_clean_import_control(tmp_path: Path) -> None:
    result = _probe_fixture(tmp_path, "")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["forbidden_modules"] == []


@pytest.mark.parametrize("module", ["pandas", "xarray"])
def test_subprocess_gate_fails_when_legacy_import_is_reintroduced(
    tmp_path: Path, module: str
) -> None:
    result = _probe_fixture(tmp_path, f"import {module}\n")
    assert result.returncode != 0
    assert "Forbidden imported modules" in result.stderr
    assert module in result.stderr


@pytest.mark.parametrize("module", ["pandas", "xarray"])
def test_subprocess_gate_rejects_installed_legacy_distribution(
    tmp_path: Path, module: str
) -> None:
    result = _probe_fixture(tmp_path, "", distribution=module)
    assert result.returncode != 0
    assert "Forbidden installed distributions" in result.stderr


def test_subprocess_gate_rejects_source_shadowing(tmp_path: Path) -> None:
    result = _probe_fixture(tmp_path, "", expected=tmp_path / "different-prefix")
    assert result.returncode != 0
    assert "Source-shadowed package" in result.stderr


def test_no_extras_tox_lane_has_no_producer_or_optional_dependencies() -> None:
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    assert "pandas-free" in config["tox"]["env_list"].split()
    lane = config["testenv:pandas-free"]
    for field in ("extras", "dependency_groups", "deps"):
        assert lane[field].strip() == ""
    assert "--probe" in lane["commands"]
    assert "--expected-prefix {env_dir}" in lane["commands"]
    assert "python -I" in lane["commands"]


def test_required_lint_runs_the_exemption_free_source_gate() -> None:
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    assert (
        "scripts/check_pandas_free.py --source voiage"
        in config["testenv:lint"]["commands"]
    )


def test_hosted_ci_runs_the_real_no_extras_lane() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["pandas-free-runtime"]
    assert not job.get("continue-on-error", False)
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "tox run -e pandas-free" in commands
    assert job["strategy"]["matrix"]["os"] == [
        "ubuntu-24.04",
        "macos-14",
        "windows-2025",
    ]
