"""Source and behavior controls for the split current Polars providers."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts.check_current_spack_providers import validate

ROOT = Path(__file__).parents[1]


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "packaging/current", root / "packaging/current")
    sources = tmp_path / "sources"
    sources.mkdir()
    manifest = root / "packaging/current/polars-provider-sources.json"
    packet = json.loads(manifest.read_text())
    for name, row in packet["providers"].items():
        native = name.endswith("runtime-32")
        project = {
            "name": "polars-runtime-32" if native else "polars",
            "requires-python": row["requires_python"],
            "license": {"file": "LICENSE"},
        }
        project.update({"dynamic": ["version"]} if native else {"version": "2.1.0"})
        project["dependencies"] = row["run_requires"]
        # JSON strings/arrays use the same literal syntax as these TOML fields.
        metadata = (
            "[project]\n"
            + "".join(
                f"{k} = {json.dumps(v)}\n" for k, v in project.items() if k != "license"
            )
            + '[project.license]\nfile="LICENSE"\n[build-system]\nrequires='
            + json.dumps(row["build_requires"])
            + "\n"
        )
        metadata += "build-backend=" + json.dumps(row["build_backend"]) + "\n"
        files = {
            "pyproject.toml": metadata.encode(),
            "LICENSE": b"MIT fixture license\n",
        }
        if native:
            files["pyproject.toml"] += (
                "[tool.maturin]\nmanifest-path="
                + json.dumps(row["maturin_manifest"])
                + "\n"
            ).encode()
            files["rust-toolchain.toml"] = (
                "[toolchain]\nchannel=" + json.dumps(row["rust_toolchain"]) + "\n"
            ).encode()
            files[row["maturin_manifest"]] = (
                b'[package]\nversion="2.1.0"\n[features]\ndefault=["full","nightly"]\n'
            )
        row["license_sha256"] = hashlib.sha256(files["LICENSE"]).hexdigest()
        archive = sources / row["source"]["filename"]
        with tarfile.open(archive, "w:gz") as output:
            for local, data in files.items():
                member = tarfile.TarInfo("source/" + local)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
        old = row["source"]["sha256"]
        row["source"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        recipe = root / f"packaging/current/spack/packages/{name}/package.py"
        recipe.write_text(recipe.read_text().replace(old, row["source"]["sha256"]))
    manifest.write_text(json.dumps(packet))
    return root, sources


def test_source_checked_providers_do_not_claim_native_acceptance(
    candidate: tuple[Path, Path],
) -> None:
    report = validate(*candidate)
    assert report["source_declarations_qualified"] is True
    assert report["providers_checked"] == 2
    assert report["native_solver_executed"] is False
    assert report["native_build_executed"] is False
    assert report["installed_module_acceptance"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "dependency",
        "compiler",
        "features",
        "version",
        "missing",
        "pair",
        "binding",
        "sdk-contract",
        "conditional",
    ],
)
def test_recipe_and_source_drift_is_rejected(
    candidate: tuple[Path, Path], damage: str
) -> None:
    root, sources = candidate
    recipe = root / "packaging/current/spack/packages/py-polars-runtime-32/package.py"
    if damage == "checksum":
        archive = next(sources.glob("polars_runtime*"))
        archive.write_bytes(archive.read_bytes() + b"changed")
    elif damage == "missing":
        next(sources.glob("polars_runtime*")).unlink()
    elif damage == "dependency":
        recipe.write_text(
            recipe.read_text() + '\n    depends_on("py-unrequested", type="run")\n'
        )
    elif damage == "compiler":
        recipe.write_text(
            recipe.read_text().replace("rust@=nightly-2026-09-01", "rust@nightly")
        )
    elif damage == "features":
        recipe.write_text(
            recipe.read_text()
            + '        env.set("MATURIN_PEP517_ARGS", "--no-default-features")\n'
        )
    elif damage == "version":
        recipe.write_text(recipe.read_text().replace('"2.1.0",', '"2.2.0",'))
    elif damage == "pair":
        manifest = root / "packaging/current/polars-provider-sources.json"
        packet = json.loads(manifest.read_text())
        packet["providers"] = {}
        manifest.write_text(json.dumps(packet))
    elif damage == "binding":
        recipe.write_text(
            recipe.read_text().replace("prefix.bin.rustc", "prefix.bin.rustc_stable")
        )
    elif damage == "sdk-contract":
        manifest = root / "packaging/current/polars-provider-sources.json"
        packet = json.loads(manifest.read_text())
        dependencies = packet["providers"]["py-polars-runtime-32"]["spack_dependencies"]
        dependencies["rust@nightly"] = dependencies.pop("rust@=nightly-2026-09-01")
        manifest.write_text(json.dumps(packet))
        recipe.write_text(
            recipe.read_text().replace("rust@=nightly-2026-09-01", "rust@nightly")
        )
    elif damage == "conditional":
        recipe.write_text(
            recipe.read_text().replace(
                '    depends_on("rust@=nightly-2026-09-01", type="build")',
                '    if False:\n        depends_on("rust@=nightly-2026-09-01", type="build")',
            )
        )
    with pytest.raises((ValueError, OSError)):
        validate(root, sources)
