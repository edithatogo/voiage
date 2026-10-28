"""Actual source-content and metadata controls for current HPC candidates."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from scripts import qualify_hpc_source
from scripts.qualify_hpc_source import qualify_source

ROOT = Path(__file__).parents[1]


def test_directory_entries_count_toward_archive_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: tuple[Path, str, str]
) -> None:
    archive = tmp_path / "directories.tar"
    with tarfile.open(archive, "w") as output:
        for number in range(3):
            member = tarfile.TarInfo(f"voiage-2.2.0/dir{number}")
            member.type = tarfile.DIRTYPE
            output.addfile(member)
    monkeypatch.setattr(qualify_hpc_source, "MAX_MEMBERS", 2)
    with pytest.raises(ValueError, match="bounded source inventory"):
        qualify_source(
            archive,
            source[0],
            source[1],
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            ROOT / "specs/dependency-runtime-policy.json",
        )


def test_cli_returns_structured_rejection(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/qualify_hpc_source.py"),
            str(tmp_path / "missing.tar.gz"),
            "--revision",
            "invalid",
            "--sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["source_archive_qualified"] is False
    assert "Traceback" not in result.stderr


@pytest.fixture
def source(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, str, str]:
    repository = tmp_path / "source"
    repository.mkdir()
    policy = json.loads((ROOT / "specs/dependency-runtime-policy.json").read_text())
    files = {
        "pyproject.toml": '[project]\nname="voiage"\nversion="2.2.0"\nrequires-python='
        + json.dumps(policy["requires_python"])
        + "\ndependencies="
        + json.dumps(policy["runtime"])
        + "\n",
        "voiage/__init__.py": "VALUE = 1\n",
        "rust/Cargo.toml": '[workspace]\nmembers=["crates/voiage-python"]\nresolver="2"\n',
        "rust/Cargo.lock": "# Frozen dependency lock\n",
        "rust/crates/voiage-python/Cargo.toml": '[package]\nname="voiage-python"\nversion="2.2.0"\nedition="2021"\n',
        "rust/crates/voiage-python/src/lib.rs": "pub const VALUE: u8 = 1;\n",
        "build_backend.py": "# Fixture backend\n",
        "README.md": "Fixture package\n",
        "LICENSE": "Fixture license\n",
    }
    files["pyproject.toml"] += (
        'readme="README.md"\nlicense-files=["LICENSE"]\n'
        '[project.optional-dependencies]\ndemo=["packaging>=26"]\n'
        '[build-system]\nrequires=["maturin>=1.9"]\n'
        'build-backend="build_backend"\nbackend-path=["."]\n'
    )
    files["rust/Cargo.toml"] += '[workspace.package]\nversion="2.2.0"\n'
    if getattr(request, "param", False):
        files["pyproject.toml"] = (
            files["pyproject.toml"].replace('version="2.2.0"', 'dynamic=["version"]')
            + '\n[tool.maturin]\nmanifest-path="rust/crates/voiage-python/Cargo.toml"\n'
        )
    for name, text in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git = shutil.which("git")
    assert git
    for args in [
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "fixture",
        ],
    ]:
        subprocess.run(
            [git, "-C", str(repository), *args], check=True, capture_output=True
        )
    revision = subprocess.check_output(
        [git, "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        [git, "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    return repository, revision, tree


def archive_for(
    tmp_path: Path, source: tuple[Path, str, str], damage: str = ""
) -> Path:
    repository, revision, tree = source
    entries = {
        p.relative_to(repository).as_posix(): p.read_bytes()
        for p in repository.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(repository).parts
    }
    provenance = f"revision={revision}\ntree={tree}\nclean=true\n".encode()
    entries["rust/crates/voiage-python/source-provenance.txt"] = provenance
    policy = json.loads((ROOT / "specs/dependency-runtime-policy.json").read_text())
    entries["PKG-INFO"] = (
        "Metadata-Version: 2.4\nName: voiage\nVersion: 2.2.0\nRequires-Python: "
        + policy["requires_python"]
        + "\n"
        + "".join("Requires-Dist: " + dep + "\n" for dep in policy["runtime"])
        + 'Requires-Dist: packaging>=26; extra == "demo"\n'
    ).encode()
    if damage == "metadata-version":
        entries["PKG-INFO"] = entries["PKG-INFO"].replace(
            b"Version: 2.2.0", b"Version: 9.0.0"
        )
    elif damage == "workspace":
        entries["rust/Cargo.toml"] = entries["rust/Cargo.toml"].replace(
            b'resolver="2"', b'resolver="1"'
        )
    elif damage == "metadata":
        entries["PKG-INFO"] = entries["PKG-INFO"].replace(
            b">=3.15.1,<3.16", b">=3.12,<3.13"
        )
    elif damage == "provenance":
        entries["rust/crates/voiage-python/source-provenance.txt"] = provenance.replace(
            revision.encode(), b"0" * 40
        )
    elif damage == "payload":
        entries["voiage/__init__.py"] = b"VALUE = 999\n"
    elif damage == "missing-runtime":
        del entries["voiage/__init__.py"]
    elif damage == "missing-native":
        del entries["rust/crates/voiage-python/src/lib.rs"]
    elif damage == "missing-backend":
        del entries["build_backend.py"]
    elif damage == "missing-readme":
        del entries["README.md"]
    elif damage == "missing-license":
        del entries["LICENSE"]
    elif damage == "marked-dependency":
        entries["PKG-INFO"] += (
            b'Requires-Dist: unexpected-package>=1; python_version >= "3.0"\n'
        )
    elif damage == "undeclared-extra":
        entries["PKG-INFO"] += (
            b'Requires-Dist: unexpected-package>=1; extra == "surprise"\n'
        )
    elif damage == "missing-extra":
        entries["PKG-INFO"] = entries["PKG-INFO"].replace(
            b'Requires-Dist: packaging>=26; extra == "demo"\n', b""
        )
    elif damage == "injected-build-hook":
        entries["build_backend.py"] = b"raise RuntimeError('injected build hook')\n"
    elif damage == "runtime":
        entries["pyproject.toml"] = entries["pyproject.toml"].replace(
            b">=3.15.1,<3.16", b">=3.12,<3.13"
        )
    elif damage == "dependency":
        entries["pyproject.toml"] += (
            b'\n[project.optional-dependencies]\nlegacy=["pandas>=3"]\n'
        )
    path = tmp_path / f"{damage or 'good'}.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        for name, data in entries.items():
            member = tarfile.TarInfo("voiage-2.2.0/" + name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
        if damage == "traversal":
            member = tarfile.TarInfo("voiage-2.2.0/../../escape")
            output.addfile(member, io.BytesIO(b""))
        elif damage == "symlink":
            member = tarfile.TarInfo("voiage-2.2.0/linked")
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            output.addfile(member)
        elif damage == "duplicate":
            member = tarfile.TarInfo("voiage-2.2.0/pyproject.toml")
            member.size = len(entries["pyproject.toml"])
            output.addfile(member, io.BytesIO(entries["pyproject.toml"]))
    return path


def test_real_git_bound_source_is_qualified_without_native_claims(
    tmp_path: Path, source: tuple[Path, str, str]
) -> None:
    archive = archive_for(tmp_path, source)
    result = qualify_source(
        archive,
        source[0],
        source[1],
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        ROOT / "specs/dependency-runtime-policy.json",
    )
    assert result["source_archive_qualified"] is True
    assert result["native_build_executed"] is False
    assert result["installed_module_acceptance"] is False
    assert result["runtime_files_verified"] == 1
    assert (
        result["runtime_policy_sha256"]
        == hashlib.sha256(
            (ROOT / "specs/dependency-runtime-policy.json")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "metadata",
        "metadata-version",
        "workspace",
        "provenance",
        "payload",
        "missing-runtime",
        "missing-native",
        "missing-backend",
        "missing-readme",
        "missing-license",
        "marked-dependency",
        "undeclared-extra",
        "missing-extra",
        "injected-build-hook",
        "runtime",
        "dependency",
        "traversal",
        "symlink",
        "duplicate",
    ],
)
def test_forged_or_unsafe_source_fails(
    tmp_path: Path, source: tuple[Path, str, str], damage: str
) -> None:
    archive = archive_for(tmp_path, source, damage)
    checksum = (
        "0" * 64
        if damage == "digest"
        else hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        qualify_source(
            archive,
            source[0],
            source[1],
            checksum,
            ROOT / "specs/dependency-runtime-policy.json",
        )


@pytest.mark.parametrize("source", [True], indirect=True)
def test_cargo_bound_dynamic_version_is_qualified(
    tmp_path: Path, source: tuple[Path, str, str]
) -> None:
    archive = archive_for(tmp_path, source)
    result = qualify_source(
        archive,
        source[0],
        source[1],
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        ROOT / "specs/dependency-runtime-policy.json",
    )
    assert result["package_version"] == "2.2.0"
