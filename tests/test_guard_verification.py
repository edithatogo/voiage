"""Exercise verification watchdog termination using real isolated processes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.guard_verification import clear_temporary_directory

GUARD = Path(__file__).resolve().parents[1] / "scripts/guard_verification.py"


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
