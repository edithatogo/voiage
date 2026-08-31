"""The paper replay resolves declared archives rather than mutable root files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_paper_reproduction", ROOT / "scripts/verify_paper_reproduction.py"
)
assert SPEC is not None
assert SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def manifest() -> dict:
    return json.loads((ROOT / "paper/reproduction-manifest.json").read_text())


@pytest.fixture(autouse=True)
def source_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply source objects for unit checks that also run in shallow CI clones.

    The standalone frozen replay validates these against actual Git history.
    """
    current = manifest()
    source = current["source_reference"]

    def read_object(command: list[str], **kwargs: object) -> bytes | str:
        reference = command[-1]
        if reference == source + "^{commit}":
            return source + "\n"
        if reference == source + ":paper/reproduction-manifest.json":
            return (ROOT / current["historical_receipt"]["path"]).read_bytes()
        for name, local in (
            ("lockfile", "uv.lock"),
            ("project_file", "pyproject.toml"),
        ):
            if reference == source + ":" + local:
                return (ROOT / current[name]["path"]).read_bytes()
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(VERIFIER.subprocess, "check_output", read_object)


def test_current_manifest_resolves_exact_source_and_preserves_original_receipt() -> (
    None
):
    current = manifest()
    VERIFIER.verify_manifest(ROOT, current)
    original_path = VERIFIER.bound_file(ROOT, current["historical_receipt"])
    expected = original_path.read_bytes()
    assert current["historical_receipt"]["sha256"] == (
        "aa68d3195016842b6f8e60051ce9a28f7639c5b10dc1b8e2ce06d139090d90ce"
    )
    assert (
        hashlib.sha256(expected).hexdigest() == current["historical_receipt"]["sha256"]
    )
    assert json.loads(expected)["source_reference"] == "v2.0.0"
    assert current["source_reference"] != "v2.0.0"


@pytest.mark.parametrize("field", ["lockfile", "project_file", "historical_receipt"])
def test_replay_rejects_missing_or_tampered_declared_artifacts(field: str) -> None:
    current = manifest()
    current[field]["path"] = "missing-reproduction-artifact"
    with pytest.raises(ValueError, match="Missing declared"):
        VERIFIER.verify_manifest(ROOT, current)
    current = manifest()
    current[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        VERIFIER.verify_manifest(ROOT, current)


@pytest.mark.parametrize("path", ["../outside", "/outside"])
def test_replay_rejects_archive_path_escape(path: str) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        VERIFIER.bound_file(ROOT, {"path": path, "sha256": "0" * 64})


@pytest.mark.parametrize("source", ["v2.0.0", "main", "--help", "0" * 40])
def test_replay_rejects_unresolvable_or_mutable_source(source: str) -> None:
    current = manifest()
    current["source_reference"] = source
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        VERIFIER.verify_manifest(ROOT, current)


@pytest.mark.parametrize("field", ["lockfile", "project_file"])
def test_replay_rejects_digest_valid_file_from_wrong_source(field: str) -> None:
    current = manifest()
    alternate = ROOT / "README.md"
    current[field] = {
        "path": "README.md",
        "sha256": hashlib.sha256(alternate.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="exact source commit"):
        VERIFIER.verify_manifest(ROOT, current)


def test_declared_archive_is_independent_of_current_root_lock(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    shutil.copyfile(ROOT / manifest()["lockfile"]["path"], archive / "uv.lock")
    descriptor = dict(manifest()["lockfile"], path="archive/uv.lock")
    (tmp_path / "uv.lock").write_text("current dependency graph changed")
    assert VERIFIER.bound_file(tmp_path, descriptor) == archive / "uv.lock"
    (archive / "uv.lock").write_text("tampered historical bytes")
    with pytest.raises(ValueError, match="digest mismatch"):
        VERIFIER.bound_file(tmp_path, descriptor)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generator", "other.py"),
        ("replay_extras", []),
        ("schema_version", "voiage.paper.reproduction.v1"),
    ],
)
def test_replay_rejects_undeclared_execution_contract(
    field: str, value: object
) -> None:
    current = manifest()
    current[field] = value
    with pytest.raises(ValueError):
        VERIFIER.verify_manifest(ROOT, current)


def test_failed_generator_cannot_emit_verified_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = manifest()
    commands = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        if "clone" in command:
            checkout = Path(command[-1])
            checkout.mkdir()
            for name, local in (
                ("lockfile", "uv.lock"),
                ("project_file", "pyproject.toml"),
            ):
                shutil.copyfile(ROOT / current[name]["path"], checkout / local)
        if "--frozen" in command:
            assert command[command.index("--extra") + 1] == "plotting"
            assert kwargs["cwd"] != ROOT
            assert "VIRTUAL_ENV" not in kwargs["env"]
            raise subprocess.CalledProcessError(1, command, stderr=b"output mismatch")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(VERIFIER.subprocess, "run", run)
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated-environment")
    with pytest.raises(subprocess.CalledProcessError):
        VERIFIER.replay(ROOT, ROOT / "paper/reproduction-manifest.json")
    assert any(current["source_reference"] in command for command in commands)
    assert any("--frozen" in command for command in commands)


@pytest.mark.parametrize("field", ["seeds", "inputs", "synthetic_data", "outputs"])
def test_replay_rejects_changed_scientific_contract(field: str) -> None:
    current = manifest()
    if field == "seeds":
        current[field]["bootstrap"] += 1
    elif field == "inputs":
        current[field]["probabilistic_sensitivity_analysis_draws"] += 1
    elif field == "synthetic_data":
        current[field] = False
    else:
        current[field][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scientific contract"):
        VERIFIER.verify_manifest(ROOT, current)


def test_replay_receipt_records_observed_runtime_and_selected_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = manifest()
    observed = {
        "python_version": "3.14.6",
        "python_implementation": "CPython",
        "platform": "fixture-platform",
        "machine": "fixture-machine",
    }
    query_directories = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "clone" in command:
            checkout = Path(command[-1])
            checkout.mkdir()
            for name, local in (
                ("lockfile", "uv.lock"),
                ("project_file", "pyproject.toml"),
            ):
                shutil.copyfile(ROOT / current[name]["path"], checkout / local)
            for output in current["outputs"]:
                target = checkout / output["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / output["path"], target)
        if "--frozen" in command:
            query_directories.append(kwargs["cwd"])
            assert command[command.index("--extra") + 1] == "plotting"
        if "-c" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(observed).encode(), b""
            )
        return subprocess.CompletedProcess(command, 0, b"tracked outputs match", b"")

    monkeypatch.setattr(VERIFIER.subprocess, "run", run)
    receipt = VERIFIER.replay(ROOT, ROOT / "paper/reproduction-manifest.json")
    assert receipt["runtime"] == observed
    assert receipt["replay_extras"] == ["plotting"]
    assert len(query_directories) == 2
    assert query_directories[0] == query_directories[1]
    assert query_directories[0] != ROOT
