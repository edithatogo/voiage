"""Checks for immutable HPC inputs and fail-closed build smoke receipts."""

import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

from packaging.requirements import Requirement
import pytest

from scripts import hpc_package_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]


def _easyconfig(year: str) -> dict:
    path = ROOT / f"packaging/easybuild/voiage-2.2.0-foss-{year}.eb"
    values = {}
    for node in ast.parse(path.read_text()).body:
        assert isinstance(node, ast.Assign)
        if isinstance(node.value, ast.List) and any(
            isinstance(item, ast.Name) for item in node.value.elts
        ):
            assert ast.unparse(node.value) == "[SOURCE_TAR_GZ]"
            value = ["voiage-2.2.0.tar.gz"]
        else:
            value = ast.literal_eval(node.value)
        values[node.targets[0].id] = value
    return values


@pytest.mark.parametrize("year", ["2023a", "2024a"])
def test_easyconfig_runtime_pins_satisfy_release_requirements(year: str) -> None:
    import tomllib

    config = _easyconfig(year)
    requirements = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    pins = {
        name.lower().replace("_", "-"): version
        for name, version in config["dependencies"]
    }
    for text in requirements:
        requirement = Requirement(text)
        assert pins[requirement.name.lower().replace("_", "-")] in requirement.specifier
    assert {name: pins[name] for name in smoke.RUNTIME_PINS} == smoke.RUNTIME_PINS
    assert config["version"] == "2.2.0"
    assert config["toolchain"] == {"name": "foss", "version": year}
    assert config["checksums"] == [smoke.SOURCE_SHA256]
    assert config["pip_no_index"]
    assert config["download_dep_fail"]
    assert config["sanity_pip_check"]
    assert config["moduleclass"] == "math"
    assert "voiage --help" in config["sanity_check_commands"]


def test_spack_declares_all_runtime_dependencies_and_native_import() -> None:
    text = (ROOT / "packaging/spack/package.py").read_text()
    tree = ast.parse(text)
    dependencies = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "depends_on"
    }
    assert "py-jsonschema@4.26:4" in dependencies
    assert "py-click@8.5:8" in dependencies
    assert "py-typer@0.27.2:0" in dependencies
    assert "python@3.12:3.14" in dependencies
    assert "rust@1.85:" in dependencies
    assert len(dependencies) == 15  # Python, Rust, Maturin and 12 runtime packages.
    assert smoke.SOURCE_SHA256 in text
    assert '"voiage._core"' in text


def test_changed_source_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"not the pinned source")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        smoke.extract_source(archive, tmp_path / "target")
    assert not (tmp_path / "target").exists()


def test_unsafe_tar_member_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("voiage-2.2.0/../../escaped")
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    monkeypatch.setattr(
        smoke, "SOURCE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="Unsafe"):
        smoke.extract_source(archive, tmp_path / "target")
    assert not (tmp_path / "escaped").exists()


def test_failed_smoke_writes_failure_not_hpc_success(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"wrong source")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        smoke.run_smoke(receipt, archive)
    result = json.loads(receipt.read_text())
    assert result["status"] == "failed"
    assert result["steps"] == []
    assert result["spack_build_executed"] is False
    assert result["easybuild_build_executed"] is False
