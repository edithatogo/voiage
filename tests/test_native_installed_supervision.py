"""Qualify cleanup across the native driver's actual UID and namespace boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from scripts.enter_native_namespace import preserved_environment_names

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="Requires the dedicated Linux root-supervisor qualification job",
)
def test_aggregate_stop_owns_privileged_monitor_and_detached_native_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uid = int(os.environ["VOIAGE_CONTROL_UID"])
    gid = int(os.environ["VOIAGE_CONTROL_GID"])
    tmp_path.parent.chmod(0o755)
    os.chown(tmp_path, uid, gid)
    tmp_path.chmod(0o750)
    pids = tmp_path / "native-pids.json"
    receipts = str(tmp_path / "compiler-receipts")
    monkeypatch.setenv("VOIAGE_COMPILER_RECEIPTS", receipts)
    payload = (
        "import json,os,subprocess,sys,time,psutil; from pathlib import Path; "
        f"assert os.environ.get('VOIAGE_COMPILER_RECEIPTS')=={receipts!r}; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); "
        f"Path({str(pids)!r}).write_text(json.dumps(["
        "{'pid':p,'created':psutil.Process(p).create_time()} for p in [os.getpid(),child.pid]])); "
        "time.sleep(60)"
    )
    stage = [
        "/usr/bin/sudo",
        "--non-interactive",
        f"--preserve-env={preserved_environment_names()}",
        "env",
        f"PATH={os.environ['PATH']}",
        sys.executable,
        "-m",
        "scripts.observe_native_stage",
        "--evidence",
        str(tmp_path / "stage"),
        "--workspace",
        str(tmp_path),
        "--cwd",
        str(ROOT),
        "--timeout",
        "20",
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        "--",
        sys.executable,
        "-c",
        payload,
    ]
    command = [
        sys.executable,
        "-m",
        "scripts.observe_native_driver",
        "--evidence",
        str(tmp_path / "aggregate"),
        "--workspace",
        str(tmp_path),
        "--cwd",
        str(ROOT),
        "--timeout",
        "6",
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        "--",
        sys.executable,
        "-c",
        f"import subprocess;raise SystemExit(subprocess.call({stage!r}))",
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=35, check=False
    )
    report = json.loads((tmp_path / "aggregate/resource-report.json").read_text())
    assert result.returncode == 125, result.stderr
    assert report["trigger_reason"] == "time_limit"
    assert report["cleanup_complete"] is True
    assert report["elapsed_seconds"] < 18
    for item in json.loads(pids.read_text()):
        try:
            process = psutil.Process(item["pid"])
            assert (
                process.create_time() != item["created"]
                or process.status() == psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            # A reaped process is the expected successful cleanup outcome.
            pass
