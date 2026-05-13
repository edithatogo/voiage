from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voiage import versioning
from voiage.core import gpu_acceleration as gpu


def test_versioning_helpers_cover_error_paths(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool]\nname = 'voiage'\n", encoding="utf-8")

    with pytest.raises(versioning.VersionSyncError, match="missing \\[project\\] table"):
        versioning._read_canonical_version(pyproject)

    description = tmp_path / "DESCRIPTION"
    description.write_text("Package: voiageR\n", encoding="utf-8")
    with pytest.raises(versioning.VersionSyncError, match="missing Version field"):
        versioning._read_description_version(description)

    canonical_empty = tmp_path / "pyproject-empty.toml"
    canonical_empty.write_text("[project]\nversion = ''\n", encoding="utf-8")
    with pytest.raises(versioning.VersionSyncError, match="missing project.version"):
        versioning._read_canonical_version(canonical_empty)

    json_empty = tmp_path / "empty.json"
    json_empty.write_text('{"version": ""}', encoding="utf-8")
    with pytest.raises(versioning.VersionSyncError, match="missing version"):
        versioning._read_json_version(json_empty)

    csproj = tmp_path / "Voiage.Core.csproj"
    csproj.write_text("<Project><PropertyGroup></PropertyGroup></Project>", encoding="utf-8")
    with pytest.raises(versioning.VersionSyncError, match="missing <Version>"):
        versioning._read_csproj_version(csproj)

    mismatches = [
        versioning.VersionMismatch(
            label="Python",
            path=tmp_path / "pyproject.toml",
            expected="1.2.3",
            found="1.2.0",
        )
    ]
    formatted = versioning.format_version_mismatches(mismatches, repo_root=tmp_path)
    assert "version synchronization failed" in formatted
    assert "Python" in formatted
    assert "1.2.3" in formatted


def test_versioning_load_helpers_and_version_extractors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path = tmp_path / "sample.toml"
    toml_path.write_text("version = '1.2.3'\n", encoding="utf-8")

    json_path = tmp_path / "sample.json"
    json_path.write_text('{"version": "1.2.3"}', encoding="utf-8")

    assert versioning._load_toml(toml_path)["version"] == "1.2.3"
    assert versioning._load_json(json_path)["version"] == "1.2.3"
    assert versioning._read_toml_version(toml_path) == "1.2.3"

    cargo_path = tmp_path / "Cargo.toml"
    cargo_path.write_text("[package]\nversion = '1.2.3'\n", encoding="utf-8")
    assert versioning._read_cargo_version(cargo_path) == "1.2.3"

    monkeypatch.setattr(versioning.tomllib, "load", lambda handle: [])
    with pytest.raises(versioning.VersionSyncError, match="expected TOML document object"):
        versioning._load_toml(toml_path)

    monkeypatch.setattr(versioning.json, "load", lambda handle: [])
    with pytest.raises(versioning.VersionSyncError, match="expected JSON object"):
        versioning._load_json(json_path)

    monkeypatch.undo()
    with pytest.raises(versioning.VersionSyncError, match="missing missing"):
        versioning._read_toml_version(toml_path, key_path=("missing",))

    empty_version = tmp_path / "empty.toml"
    empty_version.write_text("version = ''\n", encoding="utf-8")
    with pytest.raises(versioning.VersionSyncError, match="missing version"):
        versioning._read_toml_version(empty_version)


def test_collect_version_mismatches_reports_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nversion = '1.2.3'\n", encoding="utf-8"
    )

    with pytest.raises(versioning.VersionSyncError, match="missing manifest"):
        versioning.collect_version_mismatches(tmp_path)


def test_collect_version_mismatches_reports_drift_and_main(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nversion = '1.2.3'\n", encoding="utf-8"
    )
    (tmp_path / "bindings").mkdir()
    (tmp_path / "bindings/typescript").mkdir(parents=True)
    (tmp_path / "bindings/typescript/package.json").write_text(
        '{"version": "1.2.0"}', encoding="utf-8"
    )
    (tmp_path / "bindings/julia").mkdir(parents=True)
    (tmp_path / "bindings/julia/Project.toml").write_text(
        'version = "1.2.3"', encoding="utf-8"
    )
    (tmp_path / "bindings/rust").mkdir(parents=True)
    (tmp_path / "bindings/rust/Cargo.toml").write_text(
        "[package]\nversion = '1.2.3'\n", encoding="utf-8"
    )
    (tmp_path / "bindings/dotnet/src/Voiage.Core").mkdir(parents=True)
    (tmp_path / "bindings/dotnet/src/Voiage.Core/Voiage.Core.csproj").write_text(
        "<Project><PropertyGroup><Version>1.2.3</Version></PropertyGroup></Project>",
        encoding="utf-8",
    )
    (tmp_path / "r-package/voiageR").mkdir(parents=True)
    (tmp_path / "r-package/voiageR/DESCRIPTION").write_text(
        "Package: voiageR\nVersion: 1.2.3\n", encoding="utf-8"
    )

    canonical, mismatches = versioning.collect_version_mismatches(tmp_path)
    assert canonical == "1.2.3"
    assert len(mismatches) == 1
    assert mismatches[0].label == "TypeScript"

    assert versioning.main(["--repo-root", str(tmp_path)]) == 1
    assert "version synchronization failed" in capsys.readouterr().err


def test_validate_version_sync_and_main_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nversion = '1.2.3'\n", encoding="utf-8"
    )
    (tmp_path / "bindings/typescript").mkdir(parents=True)
    (tmp_path / "bindings/typescript/package.json").write_text(
        '{"version": "1.2.3"}', encoding="utf-8"
    )
    (tmp_path / "bindings/julia").mkdir(parents=True)
    (tmp_path / "bindings/julia/Project.toml").write_text(
        'version = "1.2.3"', encoding="utf-8"
    )
    (tmp_path / "bindings/rust").mkdir(parents=True)
    (tmp_path / "bindings/rust/Cargo.toml").write_text(
        "[package]\nversion = '1.2.3'\n", encoding="utf-8"
    )
    (tmp_path / "bindings/dotnet/src/Voiage.Core").mkdir(parents=True)
    (tmp_path / "bindings/dotnet/src/Voiage.Core/Voiage.Core.csproj").write_text(
        "<Project><PropertyGroup><Version>1.2.3</Version></PropertyGroup></Project>",
        encoding="utf-8",
    )
    (tmp_path / "r-package/voiageR").mkdir(parents=True)
    (tmp_path / "r-package/voiageR/DESCRIPTION").write_text(
        "Package: voiageR\nVersion: 1.2.3\n", encoding="utf-8"
    )

    canonical, mismatches = versioning.validate_version_sync(tmp_path)
    assert canonical == "1.2.3"
    assert mismatches == []

    assert versioning.main(["--repo-root", str(tmp_path)]) == 0
    assert "validated version synchronization" in capsys.readouterr().out


def test_gpu_acceleration_auto_detect_and_invalid_backend_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    monkeypatch.setattr(gpu, "CUPY_AVAILABLE", False)
    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)
    assert gpu.get_gpu_backend() == "none"
    assert gpu.is_gpu_available() is False

    arr = np.array([1.0, 2.0], dtype=float)
    assert np.array_equal(gpu.array_to_cpu(arr), arr)

    with pytest.raises(ValueError, match="Unknown backend"):
        gpu.array_to_gpu(arr, backend="invalid")

    with pytest.raises(ValueError, match="Unknown backend"):
        gpu.array_to_cpu(arr, backend="invalid")

    def _double(value: int) -> int:
        return value * 2

    monkeypatch.setattr(gpu, "get_gpu_backend", lambda: "none")
    assert gpu.gpu_jit_compile(_double)(4) == 8
    assert gpu.gpu_vectorize(_double)(4) == 8
    assert gpu.gpu_parallelize(_double)(4) == 8

    monkeypatch.setattr(gpu, "JAX_AVAILABLE", False)
    assert gpu.gpu_jit_compile(_double, backend="jax")(4) == 8
    assert gpu.gpu_vectorize(_double, backend="jax")(4) == 8
    assert gpu.gpu_parallelize(_double, backend="jax")(4) == 8

    monkeypatch.setattr(gpu, "TORCH_AVAILABLE", False)
    assert gpu.gpu_jit_compile(_double, backend="torch")(4) == 8

    monkeypatch.setattr(
        gpu, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    assert gpu.get_gpu_backend() == "none"
