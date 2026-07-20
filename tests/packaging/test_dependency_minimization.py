"""Contracts for the minimal supported Python dependency surface."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

from voiage.core import memory_optimization
from voiage.exceptions import OptionalDependencyError

ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_probe(probe: str) -> subprocess.CompletedProcess[str]:
    """Run a clean-install probe while explicitly exposing the source tree."""
    isolated_probe = f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n{probe}"
    return subprocess.run(
        [sys.executable, "-c", isolated_probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def _dependency_names(requirements: list[str]) -> set[str]:
    return {
        requirement.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        for requirement in requirements
    }


def test_base_dependencies_match_stable_api_policy() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = _dependency_names(metadata["project"]["dependencies"])

    assert {
        "defusedxml",
        "matplotlib",
        "numpyro",
        "psutil",
        "seaborn",
        "statsmodels",
    }.isdisjoint(base)
    # Stable schemas are Xarray-backed, and Xarray itself requires Pandas.
    assert {"pandas", "xarray"} <= base
    assert "jax" not in base


def test_optional_dependencies_expose_only_real_runtime_features() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    assert extras["ecosystem"] == ["defusedxml>=0.7.1,<1.0"]
    assert extras["plotting"] == ["matplotlib>=3.4,<4.0"]
    assert extras["jax"] == [
        "jax>=0.4.33,<0.6; python_version<'3.11'",
        "jax>=0.7.1,<0.8; python_version>='3.11'",
    ]
    assert _dependency_names(extras["performance"]) == {"psutil"}
    assert "defusedxml" in _dependency_names(extras["ci"])
    assert all(
        "seaborn" not in _dependency_names(requirements)
        for requirements in extras.values()
    )


def test_base_install_imports_and_runs_numpy_without_jax() -> None:
    """Model a clean wheel environment where the JAX extra is absent."""
    probe = r"""
import importlib.abc
import sys

class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ModuleNotFoundError("JAX intentionally absent from base install")
        return None

sys.meta_path.insert(0, BlockJax())

import numpy as np
import voiage
from voiage.backends import NumpyBackend, get_backend

assert voiage.evpi(np.array([[1.0, 3.0], [4.0, 2.0]])) == 1.0
assert isinstance(get_backend("numpy"), NumpyBackend)
assert "health_economics" in voiage.__all__
assert "multi_domain" in voiage.__all__

for name in ("health_economics", "multi_domain"):
    try:
        getattr(voiage, name)
    except voiage.exceptions.OptionalDependencyError as error:
        assert "voiage[jax]" in str(error)
    else:
        raise AssertionError(f"{name} did not produce a governed JAX diagnostic")
"""
    result = _run_isolated_probe(probe)
    assert result.returncode == 0, result.stderr


def test_base_import_preserves_lazy_ecosystem_and_plotting_contracts() -> None:
    """A clean base install must not import optional ecosystem or plotting stacks."""
    probe = r"""
import importlib.abc
import sys

BLOCKED = {"defusedxml", "matplotlib"}

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"{root} intentionally absent from base install", name=root
            )
        return None

sys.meta_path.insert(0, BlockOptional())

import numpy as np
import voiage

assert "voiage.ecosystem_integration" not in sys.modules
assert "voiage.plot" not in sys.modules
assert voiage.evpi(np.array([[1.0, 3.0], [4.0, 2.0]])) == 1.0
for name in ("ecosystem_integration", "HeomlRunBundle", "load_heoml_run_bundle"):
    assert name in voiage.__all__
    try:
        getattr(voiage, name)
    except voiage.exceptions.OptionalDependencyError as error:
        assert "voiage[ecosystem]" in str(error)
    else:
        raise AssertionError(f"{name} did not produce an ecosystem-extra diagnostic")

assert "plot" in voiage.__all__
plot = voiage.plot
assert "plot_ceaf" in plot.__all__
try:
    plot.plot_evpi_vs_wtp([1.0], [100.0])
except voiage.exceptions.PlottingError as error:
    assert "voiage[plotting]" in str(error)
else:
    raise AssertionError("plotting did not produce a plotting-extra diagnostic")
"""
    result = _run_isolated_probe(probe)
    assert result.returncode == 0, result.stderr


def test_memory_optimizer_names_performance_extra_when_psutil_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_optimization, "psutil", None)

    with pytest.raises(
        OptionalDependencyError, match=r"pip install 'voiage\[performance\]'"
    ):
        memory_optimization.MemoryOptimizer()

    optimizer = memory_optimization.MemoryOptimizer(memory_limit_mb=128)
    with pytest.raises(
        OptionalDependencyError, match=r"pip install 'voiage\[performance\]'"
    ):
        optimizer.get_memory_usage()
