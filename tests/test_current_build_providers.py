"""Reject wheel-era runtime caps and changes to native build prerequisites."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts.check_current_build_providers import validate

ROOT = Path(__file__).parents[1]


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "packaging/current", root / "packaging/current")
    sources = tmp_path / "sources"
    sources.mkdir()
    path = root / "packaging/current/runtime-build-provider-sources.json"
    packet = json.loads(path.read_text())
    for name, row in packet["providers"].items():
        files = {key: (key + " fixture license").encode() for key in row["licenses"]}
        if name == "py-cython":
            files["PKG-INFO"] = (
                b"Name: Cython\nVersion: 3.3.0\nRequires-Python: >=3.9\nLicense: Apache-2.0\n\n"
            )
            files["setup.py"] = b"PYTHON_REQUIRES = '>=3.9'\n"
        else:
            files["pyproject.toml"] = (
                b'[build-system]\nrequires=["maturin>=1.10,<2"]\nbuild-backend="maturin"\n[project]\nname="pydantic_core"\nrequires-python=">=3.9"\nlicense="MIT"\ndependencies=["typing-extensions>=4.14.1"]\n'
            )
            files["Cargo.toml"] = (
                b'[package]\nname="pydantic-core"\nversion="2.46.4"\nrust-version="1.88"\nlicense="MIT"\n[dependencies.pyo3]\nversion="0.28"\n[profile.release]\nlto="fat"\ncodegen-units=1\nstrip=true\n'
            )
        row["licenses"] = {
            key: hashlib.sha256(files[key]).hexdigest() for key in row["licenses"]
        }
        archive = sources / row["source"]["filename"]
        with tarfile.open(archive, "w:gz") as output:
            for key, content in files.items():
                member = tarfile.TarInfo("source/" + key)
                member.size = len(content)
                output.addfile(member, io.BytesIO(content))
        old = row["source"]["sha256"]
        row["source"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        recipe = root / f"packaging/current/spack/packages/{name}/package.py"
        recipe.write_text(recipe.read_text().replace(old, row["source"]["sha256"]))
    path.write_text(json.dumps(packet))
    return root, sources


def test_native_prerequisite_sources_do_not_claim_installed_acceptance(
    candidate: tuple[Path, Path],
) -> None:
    report = validate(*candidate)
    assert report["source_declarations_qualified"] is True
    assert report["providers_checked"] == 2
    assert report["native_build_executed"] is False
    assert report["full_hpc_matrix_qualified"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "old-python",
        "rust-floor",
        "regeneration",
        "bypass",
        "conditional",
        "missing-hook",
        "source",
        "missing",
    ],
)
def test_prerequisite_drift_is_rejected(
    candidate: tuple[Path, Path], damage: str
) -> None:
    root, sources = candidate
    recipe = root / "packaging/current/spack/packages/py-pydantic-core/package.py"
    if damage == "old-python":
        recipe.write_text(
            recipe.read_text()
            + '\n    depends_on("python@:3.14", type=("build", "run"))\n'
        )
    elif damage == "rust-floor":
        recipe.write_text(recipe.read_text().replace("rust@1.88:1", "rust@1.75:1"))
    elif damage == "regeneration":
        recipe = root / "packaging/current/spack/packages/py-cython/package.py"
        recipe.write_text(
            recipe.read_text().replace(
                'env.set("CYTHON_FORCE_REGEN", "1")',
                'env.set("CYTHON_FORCE_REGEN", "0")',
            )
        )
    elif damage == "bypass":
        recipe.write_text(
            recipe.read_text()
            + '        env.set("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")\n'
        )
    elif damage == "missing-hook":
        recipe.write_text(
            recipe.read_text().replace(
                "def setup_build_environment", "def inactive_build_environment"
            )
        )
    elif damage == "conditional":
        recipe.write_text(recipe.read_text() + "\n    if False:\n        pass\n")
    elif damage == "source":
        next(sources.iterdir()).write_bytes(b"changed")
    elif damage == "missing":
        next(sources.iterdir()).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(root, sources)
