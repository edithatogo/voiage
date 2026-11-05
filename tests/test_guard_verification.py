"""Exercise verification watchdog termination using real isolated processes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import guard_verification as guard
from scripts.guard_verification import clear_temporary_directory

GUARD = Path(__file__).resolve().parents[1] / "scripts/guard_verification.py"


def test_workload_identity_uses_explicit_source_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("uv.lock", "pyproject.toml", "tox.ini"):
        (tmp_path / name).write_text(f"fixture-{name}")
    calls = []

    def fake_git(arguments: list[str], *, text: bool) -> str:
        assert text
        calls.append(arguments)
        return "fixture-output\n"

    monkeypatch.setattr(guard.subprocess, "check_output", fake_git)
    args = argparse.Namespace(
        source_root=tmp_path,
        workload_id="source-root-test",
        cache_state="unknown",
        identity_file=None,
    )
    identity = guard.workload_identity(args, ["example"])
    assert all(call[1:3] == ["-C", str(tmp_path)] for call in calls)
    assert (
        identity["configuration_sha256"]["uv.lock"]
        == hashlib.sha256(b"fixture-uv.lock").hexdigest()
    )


def test_guard_cli_binds_selected_checkout_when_script_is_elsewhere(
    tmp_path: Path,
) -> None:
    source_root = GUARD.parent.parent
    copied_script = tmp_path / "other" / "guard.py"
    copied_script.parent.mkdir()
    shutil.copyfile(GUARD, copied_script)
    result = subprocess.run(
        [
            sys.executable,
            str(copied_script),
            "--source-root",
            str(source_root),
            "--work-dir",
            str(tmp_path / "report"),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        check=False,
        timeout=20,
    )
    assert result.returncode == 0
    report = json.loads((tmp_path / "report/resource-report.json").read_text())
    expected_head = subprocess.check_output(
        ["/usr/bin/git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert report["workload"]["source_root"] == str(source_root.resolve())
    assert report["workload"]["source_head"] == expected_head


def test_disk_samples_deduplicate_filesystems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    free = iter([100, 60, 80])
    monkeypatch.setattr(
        guard.shutil,
        "disk_usage",
        lambda _: type("Usage", (), {"free": next(free), "total": 200})(),
    )
    samples = guard.DiskMeasurements([tmp_path, nested])
    samples.sample()
    samples.sample()
    samples.sample()
    (record,) = samples.records.values()
    assert len(record["paths"]) == 2
    assert record["start_free_bytes"] == 100
    assert record["min_free_bytes"] == 60
    assert record["end_free_bytes"] == 80
    assert record["samples"] == 3


def test_artifact_sizes_do_not_follow_links_or_count_hardlinks_twice(
    tmp_path: Path,
) -> None:
    import os

    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "data").write_bytes(b"x" * 8192)
    os.link(root / "data", root / "alias")
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    report = guard.artifact_size(root)
    assert report["regular_file_bytes"] == 8192
    assert report["allocated_bytes"] >= 8192
    assert report["symlinks_skipped"] == 1
    assert guard.artifact_size(root / "missing")["status"] == "absent"


def test_short_job_records_disk_identity_and_before_after_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "outputs"
    code = f"from pathlib import Path; p=Path({str(artifacts)!r}); p.mkdir(); (p/'data').write_bytes(b'x'*8192)"
    result = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--work-dir",
            str(tmp_path / "report"),
            "--workload-id",
            "short-test",
            "--cache-state",
            "cold",
            "--artifact-dir",
            str(artifacts),
            "--",
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        timeout=15,
    )
    assert result.returncode == 0
    report = json.loads((tmp_path / "report/resource-report.json").read_text())
    assert {
        "peak_rss_kib",
        "reason",
        "returncode",
        "elapsed_seconds",
        "temporary_directory",
    } <= report.keys()
    assert report["workload"]["id"] == "short-test"
    assert report["workload"]["cache_state"] == "cold"
    assert report["disk"]["artifacts_before"][0]["status"] == "absent"
    assert report["disk"]["artifacts_after"][0]["regular_file_bytes"] == 8192
    assert all(item["samples"] >= 2 for item in report["disk"]["filesystems"])
    assert report["disk"]["physical_capacity_qualified"] is False


def test_monitor_failure_still_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", [str(GUARD), "--work-dir", str(tmp_path), "--", "unused"]
    )

    def fail(_: object) -> object:
        raise OSError("disk unavailable")

    monkeypatch.setattr(guard.shutil, "disk_usage", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        guard.main()
    report = json.loads((tmp_path / "resource-report.json").read_text())
    assert report["reason"] == "interrupted_or_monitor_failed"


def test_free_space_drop_stops_job_and_preserves_temporary_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", [str(GUARD), "--work-dir", str(tmp_path), "--", "unused"]
    )
    observations = iter([4 * 1024**3, 1, 1])
    monkeypatch.setattr(guard.DiskMeasurements, "sample", lambda _: next(observations))
    stopped = []
    monkeypatch.setattr(guard, "stop_group", stopped.append)
    monkeypatch.setattr(guard, "group_rss_kib", lambda _: 0)

    class Child:
        pid = 12345

        def poll(self) -> None:
            return None

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(guard.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(guard, "workload_identity", lambda *_: {})
    assert guard.main() == 125
    report = json.loads((tmp_path / "resource-report.json").read_text())
    assert report["reason"] == "free_space_limit"
    assert stopped == [12345, 12345]
    private = Path(report["temporary_directory"])
    assert private.is_dir()
    private.rmdir()


def test_terminal_monitor_failure_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", [str(GUARD), "--work-dir", str(tmp_path), "--", "unused"]
    )
    calls = 0

    def sample(_: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("terminal sample failed")
        return 4 * 1024**3

    monkeypatch.setattr(guard.DiskMeasurements, "sample", sample)
    monkeypatch.setattr(guard, "stop_group", lambda _: None)
    monkeypatch.setattr(guard, "workload_identity", lambda *_: {})

    class Child:
        pid = 12345
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(guard.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(OSError, match="terminal sample failed"):
        guard.main()
    report = json.loads((tmp_path / "resource-report.json").read_text())
    assert report["reason"] == "interrupted_or_monitor_failed"
    assert report["returncode"] == 0  # Child succeeded, but monitoring did not.
    private = Path(report["temporary_directory"])
    assert private.is_dir()
    private.rmdir()


def test_cleanup_handles_read_only_fixtures_without_following_links(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    (external / "keep.txt").write_text("untouched")
    (private / "link").symlink_to(external, target_is_directory=True)
    frozen = private / "frozen"
    frozen.mkdir()
    (frozen / "fixture.txt").write_text("fixture")
    frozen.chmod(0o555)
    clear_temporary_directory(private)
    assert not private.exists()
    assert (external / "keep.txt").read_text() == "untouched"
    assert external.stat().st_mode & 0o777 == 0o755


def test_guard_disables_detached_git_monitors_without_resetting_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "advice.detachedHead")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    output = tmp_path / "git-settings.json"
    code = (
        "import json,pathlib,subprocess; "
        "settings={k:subprocess.check_output(['/usr/bin/git','config','--get',k],"
        "text=True).strip() for k in ['core.fsmonitor','advice.detachedHead']}; "
        f"pathlib.Path({str(output)!r}).write_text(json.dumps(settings))"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--work-dir",
            str(tmp_path),
            "--interval-seconds",
            "0.05",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        timeout=15,
    )
    assert result.returncode == 0
    assert json.loads(output.read_text()) == {
        "core.fsmonitor": "false",
        "advice.detachedHead": "false",
    }


@pytest.mark.parametrize(
    ("limits", "code", "expected", "reason"),
    [
        ([], "raise SystemExit(7)", 7, "completed"),
        (
            ["--timeout-seconds", "0.1"],
            "import time; time.sleep(30)",
            125,
            "time_limit",
        ),
        (["--max-rss-mib", "1"], "import time; time.sleep(30)", 125, "memory_limit"),
        (
            ["--min-free-mib", "999999999"],
            "raise SystemExit(0)",
            125,
            "insufficient_free_space",
        ),
    ],
)
def test_guard_limits_and_exit_status(
    tmp_path: Path, limits: list[str], code: str, expected: int, reason: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--work-dir",
            str(tmp_path),
            "--interval-seconds",
            "0.05",
            *limits,
            "--",
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        timeout=15,
    )
    assert result.returncode == expected
    report = json.loads((tmp_path / "resource-report.json").read_text())
    assert report["reason"] == reason
    if reason == "memory_limit":
        assert report["peak_rss_kib"] > 1024


def test_guard_cleans_up_descendants_after_leader_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--work-dir",
            str(tmp_path),
            "--interval-seconds",
            "0.05",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        timeout=15,
    )
    assert result.returncode == 0
    child = int(pid_file.read_text())
    state = subprocess.run(
        ["/bin/ps", "-o", "stat=", "-p", str(child)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not state or state.startswith("Z"), f"Child {child} still running: {state}"


def test_tox_preserves_process_local_git_overrides() -> None:
    """Nested isolated test environments must not re-enable detached monitors."""
    import configparser

    config = configparser.ConfigParser(interpolation=None)
    config.read(GUARD.parents[1] / "tox.ini")
    forwarded = config["testenv"]["pass_env"].split()
    assert {"GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_*", "GIT_CONFIG_VALUE_*"} <= set(
        forwarded
    )
