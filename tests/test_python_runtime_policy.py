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
    assert not supported.contains("3.15.1")
    assert supported.contains("3.15.2")
    assert not supported.contains("3.15.0")
    assert not supported.contains("3.14.9")
    assert not supported.contains("3.16.0")
    assert (ROOT / ".python-version").read_text().strip() == "3.15.2"


def test_hosted_interpreters_and_python_matrices_use_qualified_patch() -> None:
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for job in workflow.get("jobs", {}).values():
            matrix = job.get("strategy", {}).get("matrix", {})
            if "python" in matrix:
                assert matrix["python"] == ["3.15.2"], path
            for step in job.get("steps", []):
                if not step.get("uses", "").startswith("actions/setup-python@"):
                    continue
                value = str(step.get("with", {}).get("python-version", ""))
                assert value in {"3.15.2", "3.15.2t", "${{ matrix.python }}"}, path


def test_tox_selects_the_qualified_patch_before_creating_environments() -> None:
    import configparser

    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    qualified = "python" + (ROOT / ".python-version").read_text().strip()
    selected = {
        config[section]["base_python"]
        for section in config.sections()
        if "base_python" in config[section]
    }
    assert selected == {"{env:VOIAGE_TOX_PYTHON:" + qualified + "}"}
    assert config["testenv"]["runner"] == "virtualenv"


def test_nox_and_license_audit_select_the_qualified_patch() -> None:
    import ast

    selected = (ROOT / ".python-version").read_text().strip()
    tree = ast.parse((ROOT / "noxfile.py").read_text())
    versions = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PYTHON_VERSIONS"
            for target in node.targets
        )
    )
    assert ast.literal_eval(versions) == [selected]
    sessions = [
        decorator
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "nox"
        and decorator.func.attr == "session"
    ]
    assert sessions
    for session in sessions:
        value = next(
            (item.value for item in session.keywords if item.arg == "python"), None
        )
        assert (isinstance(value, ast.Name) and value.id == "PYTHON_VERSIONS") or (
            isinstance(value, ast.Constant) and value.value == selected
        )
    audit = (ROOT / "scripts/run_python_license_audit.sh").read_text()
    assert 'repo_python="$repo_root/.venv/bin/python"' in audit
    assert 'uv venv --quiet --python "$repo_python"' in audit
    assert audit.count("sys.version_info[:3]") == 2
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["tool"]["taskipy"]["tasks"]["conda-build"].endswith(
        f"--python {selected}"
    )
