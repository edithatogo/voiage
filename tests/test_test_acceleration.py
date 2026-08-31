"""Safety boundaries for local selective test feedback."""

from pathlib import Path

import pytest

from scripts.test_acceleration import fingerprint, invalidate_cache, validate_targets


def fixture_repository(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_a.py").write_text("def test_a(): pass\n")
    (tmp_path / "uv.lock").write_text("version=1\n")
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs/data.json").write_text("{}")
    return tmp_path


def test_invalidation_covers_dependencies_inventory_and_data(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path)
    files = ["tests/test_a.py", "uv.lock", "specs/data.json"]
    initial = fingerprint(root, files, ["tests"], ["pytest==9"])
    # Existing Python edits are tracked by testmon, not by full-cache resets.
    (root / files[0]).write_text("def test_a(): assert True\n")
    assert fingerprint(root, files, ["tests"], ["pytest==9"]) == initial
    for path in ("uv.lock", "specs/data.json"):
        old = (root / path).read_text()
        (root / path).write_text(old + "\n")
        assert fingerprint(root, files, ["tests"], ["pytest==9"]) != initial
        (root / path).write_text(old)
    assert (
        fingerprint(root, [*files, "tests/test_new.py"], ["tests"], ["pytest==9"])
        != initial
    )
    assert fingerprint(root, files, ["tests/test_a.py"], ["pytest==9"]) != initial
    assert fingerprint(root, files, ["tests"], ["pytest==10"]) != initial


def test_bad_or_missing_manifest_discards_database(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("testmon", "testmon-wal", "testmon-shm"):
        (cache / name).write_text("stale")
    assert invalidate_cache(cache, "new")
    assert not (cache / "testmon").exists()
    (cache / "fingerprint").write_text("new")
    assert not invalidate_cache(cache, "new")
    assert invalidate_cache(cache, "changed")


@pytest.mark.parametrize(
    "target", ["../other", "/outside/test.py", "-n", "tests/test_a.py::test_a"]
)
def test_selection_rejects_options_and_escape(tmp_path: Path, target: str) -> None:
    root = fixture_repository(tmp_path)
    with pytest.raises(ValueError):
        validate_targets(root, [target])


def test_selection_accepts_existing_repository_tests(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path)
    assert validate_targets(root, ["tests/test_a.py"]) == ["tests/test_a.py"]


def test_replay_rejects_nonpublic_and_credential_bearing_requests() -> None:
    from types import SimpleNamespace

    from scripts.evaluate_http_replay import URL, permitted_request, public_response

    good = SimpleNamespace(method="GET", uri=URL, body=None, headers={})
    assert permitted_request(good) is good
    for overrides in (
        {"method": "POST"},
        {"uri": "https://private.invalid/"},
        {"body": b"sensitive"},
        {"headers": {"Authorization": "redacted"}},
    ):
        request = SimpleNamespace(method="GET", uri=URL, body=None, headers={})
        request.__dict__.update(overrides)
        with pytest.raises(ValueError):
            permitted_request(request)
    assert (
        public_response(
            {"headers": {"Set-Cookie": ["private"]}, "body": {"string": "{}"}}
        )["headers"]
        == {}
    )


@pytest.mark.parametrize(
    "profile", ["ordinary", "testmon", "gremlins", "ctrace", "sysmon"]
)
def test_profile_isolation_and_measured_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    import json
    import subprocess

    from scripts import test_acceleration as runner

    root = fixture_repository(tmp_path)
    cache = root / "cache"
    cache.mkdir()
    monkeypatch.setenv("PYTEST_ADDOPTS", "-n auto --cov")
    monkeypatch.setenv("VOIAGE_TEST_SHARD_INDEX", "1")
    monkeypatch.setenv("VOIAGE_TEST_SHARD_COUNT", "2")
    monkeypatch.setenv("COVERAGE_CORE", "sysmon")
    monkeypatch.setattr(runner.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "tests/test_a.py\nuv.lock"
    )

    def execute(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert "PYTEST_ADDOPTS" not in env
        assert "VOIAGE_TEST_SHARD_INDEX" not in env
        assert "--cov" not in command
        assert "-n" not in command
        assert env.get("COVERAGE_CORE") == (
            "ctrace"
            if profile in ("testmon", "gremlins")
            else profile
            if profile in ("ctrace", "sysmon")
            else None
        )
        if profile == "gremlins":
            assert "--gremlin-targets=voiage/mutation_policy.py" in command
            assert "tests/test_a.py" not in command
        Path(command[command.index("--junitxml") + 1]).write_text(
            '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>'
        )
        return subprocess.CompletedProcess(
            command, 0, "collected 2 items\n2 passed", ""
        )

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.run_profile(root, cache, profile, ["tests/test_a.py"]) == 0
    report = json.loads((cache / f"{profile}.json").read_text())
    assert report["counts"]["tests"] == report["collected"] == 2
    assert report["total_wall_seconds"] >= report["wall_seconds"]


@pytest.mark.parametrize(
    ("exit_code", "existing_baseline", "expected"),
    [(5, True, 0), (5, False, 5), (2, True, 2), (0, False, 1), (0, True, 0)],
)
def test_cached_empty_requires_successful_unchanged_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    existing_baseline: bool,
    expected: int,
) -> None:
    import subprocess

    from scripts import test_acceleration as runner

    root = fixture_repository(tmp_path)
    cache = root / "cache"
    cache.mkdir()
    monkeypatch.setattr(runner.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "tests/test_a.py"
    )
    signature = fingerprint(root, ["tests/test_a.py"], ["tests"], [])
    if existing_baseline:
        (cache / "fingerprint").write_text(signature)

    def execute(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert not (cache / "fingerprint").exists()
        Path(command[command.index("--junitxml") + 1]).write_text(
            '<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0"/></testsuites>'
        )
        return subprocess.CompletedProcess(command, exit_code, "", "")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.run_profile(root, cache, "testmon", ["tests"]) == expected
    assert (cache / "fingerprint").exists() == (expected == 0)


def test_timeout_removes_stale_profile_and_mutation_reports_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from scripts import test_acceleration as runner

    root = fixture_repository(tmp_path)
    cache = root / "cache"
    cache.mkdir()
    (cache / "gremlins.json").write_text('{"exit_code": 0}')
    report = root / "coverage/gremlins/gremlins.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"summary": {"total": 29, "zapped": 29}}')
    unrelated = report.parent / "keep.json"
    unrelated.write_text("unrelated")
    monkeypatch.setattr(runner.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "tests/test_a.py"
    )

    def timeout(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert not report.exists()
        assert not (cache / "gremlins.json").exists()
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_profile(root, cache, "gremlins", ["tests"])
    assert not report.exists()
    assert not (cache / "gremlins.json").exists()
    assert unrelated.read_text() == "unrelated"


@pytest.mark.parametrize("fails", [False, True])
def test_cli_holds_lock_and_releases_it_on_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    from scripts import test_acceleration as runner

    root = fixture_repository(tmp_path)
    monkeypatch.setattr(runner, "__file__", str(root / "scripts/test_acceleration.py"))
    monkeypatch.setattr(
        runner.sys, "argv", ["test_acceleration.py", "ordinary", "tests"]
    )
    lock = root / ".conductor/local/test-acceleration/running"

    def execute(repo: Path, cache: Path, profile: str, targets: list[str]) -> int:
        assert lock.is_dir()
        assert repo == root
        assert profile == "ordinary"
        assert targets == ["tests"]
        if fails:
            raise RuntimeError("runner interrupted")
        return 0

    monkeypatch.setattr(runner, "run_profile", execute)
    if fails:
        with pytest.raises(RuntimeError, match="interrupted"):
            runner.main()
    else:
        assert runner.main() == 0
    assert not lock.exists()


def test_cli_refuses_existing_lock_and_invalid_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import test_acceleration as runner

    root = fixture_repository(tmp_path)
    monkeypatch.setattr(runner, "__file__", str(root / "scripts/test_acceleration.py"))
    lock = root / ".conductor/local/test-acceleration/running"
    lock.mkdir(parents=True)
    for target in ("tests", "../escape"):
        monkeypatch.setattr(
            runner.sys, "argv", ["test_acceleration.py", "ordinary", target]
        )
        with pytest.raises(SystemExit) as error:
            runner.main()
        assert error.value.code == 2
    assert lock.exists()
