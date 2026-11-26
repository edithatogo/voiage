"""The paper replay resolves declared archives rather than mutable root files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace

from packaging.version import Version
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
    return json.loads((ROOT / VERIFIER.DEFAULT_MANIFEST).read_text())


@pytest.fixture(autouse=True)
def source_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply source objects for unit checks that also run in shallow CI clones.

    The standalone frozen replay validates these against actual Git history.
    """
    current = manifest()
    previous = json.loads(
        (ROOT / current["previous_python315_replay"]["path"]).read_text()
    )
    previous_patched = json.loads(
        (ROOT / current["previous_patched_replay"]["path"]).read_text()
    )

    def read_object(command: list[str], **kwargs: object) -> bytes | str:
        reference = command[-1]
        for profile in (current, previous_patched, previous):
            source = profile["source_reference"]
            if reference == source + "^{commit}":
                return source + "\n"
            if reference == source + ":paper/reproduction-manifest.json":
                return (ROOT / profile["historical_replay"]["path"]).read_bytes()
            if reference == source + ":paper/reproduction-manifest-python315.json":
                return (
                    ROOT / current["previous_python315_replay"]["path"]
                ).read_bytes()
            if (
                reference
                == source + ":paper/reproduction-manifest-python315-patched.json"
            ):
                return (ROOT / current["previous_patched_replay"]["path"]).read_bytes()
            for name, local in (
                ("lockfile", "uv.lock"),
                ("project_file", "pyproject.toml"),
            ):
                if reference == source + ":" + local:
                    return (ROOT / profile[name]["path"]).read_bytes()
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


def test_previous_python315_profile_remains_verifiable() -> None:
    previous = json.loads(
        (ROOT / manifest()["previous_python315_replay"]["path"]).read_text()
    )
    VERIFIER.verify_manifest(ROOT, previous)


def test_previous_patched_profile_remains_verifiable() -> None:
    previous = json.loads(
        (ROOT / manifest()["previous_patched_replay"]["path"]).read_text()
    )
    VERIFIER.verify_manifest(ROOT, previous)


@pytest.mark.parametrize(
    "field",
    [
        "lockfile",
        "project_file",
        "historical_receipt",
        "historical_replay",
        "previous_python315_replay",
        "previous_patched_replay",
    ],
)
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


@pytest.mark.parametrize(
    "field",
    [
        "lockfile",
        "project_file",
        "previous_python315_replay",
        "previous_patched_replay",
    ],
)
def test_replay_rejects_digest_valid_file_from_wrong_source(field: str) -> None:
    current = manifest()
    alternate = ROOT / "README.md"
    current[field] = {
        "path": "README.md",
        "sha256": hashlib.sha256(alternate.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="exact source commit"):
        VERIFIER.verify_manifest(ROOT, current)


def test_patched_replay_requires_the_previous_descriptor() -> None:
    current = manifest()
    del current["previous_python315_replay"]
    with pytest.raises(ValueError, match="requires its previous"):
        VERIFIER.verify_manifest(ROOT, current)


def test_python3152_replay_requires_the_previous_patched_descriptor() -> None:
    current = manifest()
    del current["previous_patched_replay"]
    with pytest.raises(ValueError, match="requires its previous patched"):
        VERIFIER.verify_manifest(ROOT, current)


@pytest.mark.parametrize("lock_path", ["uv.lock", "current-replay"])
def test_active_locks_meet_current_advisory_floors(lock_path: str) -> None:
    if lock_path == "current-replay":
        lock_path = manifest()["lockfile"]["path"]
    lock = tomllib.loads((ROOT / lock_path).read_text())
    for package, minimum in {
        "opentelemetry-api": "1.45.2",
        "pillow": "13.0.0",
        "soupsieve": "4.0.0",
        "tornado": "6.6.1",
    }.items():
        versions = [
            Version(entry["version"])
            for entry in lock["package"]
            if entry["name"] == package and "version" in entry
        ]
        assert versions
        assert all(version >= Version(minimum) for version in versions)


def test_patched_replay_archives_are_bound_by_reference_policy() -> None:
    policy = json.loads(
        (
            ROOT / "conductor/archive/pandas_free_io_20261026/reference-policy.json"
        ).read_text()
    )["files"]
    for field in ("lockfile", "project_file"):
        descriptor = manifest()[field]
        entry = policy[descriptor["path"]]
        assert entry["category"] == "frozen_historical_evidence"
        assert entry["sha256"] == descriptor["sha256"]


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
        VERIFIER.replay(ROOT, ROOT / VERIFIER.DEFAULT_MANIFEST)
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


def test_new_replay_profile_cannot_declare_an_older_runtime() -> None:
    current = manifest()
    current["supported_python_versions"] = ["3.14"]
    with pytest.raises(ValueError, match="qualified Python 3.15.2"):
        VERIFIER.verify_manifest(ROOT, current)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ("paper/reproduction-manifest.json", "Historical replay is retained"),
        (
            "paper/reproduction-manifest-python315.json",
            "retired Python runtime.*retained as evidence",
        ),
        (
            "paper/reproduction-manifest-python315-patched.json",
            "retired Python runtime.*retained as evidence",
        ),
    ],
)
def test_historical_replay_does_not_execute_a_retired_runtime(
    monkeypatch: pytest.MonkeyPatch, profile: str, message: str
) -> None:
    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("Historical replay must not start a subprocess")

    monkeypatch.setattr(VERIFIER.subprocess, "run", unexpected_run)
    monkeypatch.setattr(VERIFIER.subprocess, "check_output", unexpected_run)
    with pytest.raises(ValueError, match=message):
        VERIFIER.replay(ROOT, ROOT / profile)


def test_current_replay_rejects_an_unqualified_caller_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("An unqualified caller must not start a subprocess")

    monkeypatch.setattr(
        VERIFIER, "sys", SimpleNamespace(version_info=(3, 15, 1), executable="python")
    )
    monkeypatch.setattr(VERIFIER.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="selected Python 3.15.2 runtime"):
        VERIFIER.replay(ROOT, ROOT / VERIFIER.DEFAULT_MANIFEST)


@pytest.mark.parametrize("python_version", ["3.15.2", "3.15.1", "3.15.0", "3.14.9"])
@pytest.mark.parametrize("missing_source", [False, True])
def test_replay_receipt_records_observed_runtime_and_selected_extra(
    monkeypatch: pytest.MonkeyPatch,
    python_version: str,
    missing_source: bool,
) -> None:
    current = manifest()
    observed = {
        "python_version": python_version,
        "python_implementation": "CPython",
        "platform": "fixture-platform",
        "machine": "fixture-machine",
    }
    query_directories = []
    fetch_directories = []
    original_read = VERIFIER.subprocess.check_output

    def read_object(command: list[str], **kwargs: object) -> bytes | str:
        if missing_source and kwargs.get("stderr") == subprocess.DEVNULL:
            raise subprocess.CalledProcessError(128, command)
        return original_read(command, **kwargs)

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "fetch" in command:
            assert command[-2:] == [
                VERIFIER.SOURCE_REPOSITORY,
                current["source_reference"],
            ]
            assert kwargs["cwd"] != ROOT
            assert kwargs["timeout"] == 180
            assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
            fetch_directories.append(kwargs["cwd"])
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
            assert command[command.index("--python") + 1] == sys.executable
        if "-c" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(observed).encode(), b""
            )
        return subprocess.CompletedProcess(command, 0, b"tracked outputs match", b"")

    monkeypatch.setattr(VERIFIER.subprocess, "run", run)
    monkeypatch.setattr(VERIFIER.subprocess, "check_output", read_object)
    if python_version != "3.15.2":
        with pytest.raises(ValueError, match="did not execute the qualified"):
            VERIFIER.replay(ROOT, ROOT / VERIFIER.DEFAULT_MANIFEST)
        assert len(fetch_directories) == int(missing_source)
        return
    receipt = VERIFIER.replay(ROOT, ROOT / VERIFIER.DEFAULT_MANIFEST)
    assert receipt["runtime"] == observed
    assert receipt["replay_extras"] == ["plotting"]
    assert len(query_directories) == 2
    assert query_directories[0] == query_directories[1]
    assert query_directories[0] != ROOT
    assert len(fetch_directories) == int(missing_source)


def test_replay_rejects_source_options_before_any_git_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = manifest()
    current["source_reference"] = "--help"
    descriptor = tmp_path / "invalid.json"
    descriptor.write_text(json.dumps(current))

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid source must not reach Git")

    monkeypatch.setattr(VERIFIER.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="immutable full commit"):
        VERIFIER.replay(ROOT, descriptor)
