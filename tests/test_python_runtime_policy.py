"""Enforce the active interpreter policy independently of historical fixtures."""

from pathlib import Path
import tomllib

from packaging.specifiers import SpecifierSet
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_unavailable_torch_extra_preserves_explicit_api_diagnostic(monkeypatch) -> None:
    from voiage import metamodels

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["optional-dependencies"]["deep_learning"] == []
    disposition = config["tool"]["voiage"]["unavailable-extras"]["deep_learning"]
    assert disposition["api_retained"] is True
    assert disposition["installation_available"] is False
    monkeypatch.setattr(metamodels, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="Python 3.15.*scikit-learn"):
        metamodels.PyTorchNNMetamodel()


def test_active_package_requires_the_qualified_python_series() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    supported = SpecifierSet(project["requires-python"])
    assert supported.contains("3.15.1")
    assert not supported.contains("3.15.0")
    assert not supported.contains("3.14.9")
    assert not supported.contains("3.16.0")
    assert (ROOT / ".python-version").read_text().strip() == "3.15.1"


def test_hosted_interpreters_and_python_matrices_use_qualified_patch() -> None:
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            matrix = job.get("strategy", {}).get("matrix", {})
            if "python" in matrix:
                assert matrix["python"] == ["3.15.1"], path
            for step in job.get("steps", []):
                if not step.get("uses", "").startswith("actions/setup-python@"):
                    continue
                value = str(step.get("with", {}).get("python-version", ""))
                assert value in {"3.15.1", "3.15.1t", "${{ matrix.python }}"}, path
