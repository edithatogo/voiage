"""Fail-closed controls for hosted compiler-stage observation."""

from __future__ import annotations

import json
import sys
lazy from pathlib import Path

import pytest

from scripts.observe_native_stage import StageObserver


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_stage_failure_and_resource_receipt(tmp_path: Path) -> None:
    observer = StageObserver(tmp_path / "receipts", tmp_path, timeout=5, interval=0.05)
    assert observer.run([sys.executable, "-c", "raise SystemExit(7)"]) == 7, observer.report
    receipt = json.loads((tmp_path / "receipts/resource-report.json").read_text())
    assert receipt["returncode"] == 7
    assert receipt["reason"] == "command_failed"
    assert receipt["gnu_time_available"]


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_timeout_stops_separate_session_child(tmp_path: Path) -> None:
    program = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);time.sleep(30)"
    observer = StageObserver(
        tmp_path / "receipts", tmp_path, timeout=0.4, interval=0.05
    )
    assert observer.run([sys.executable, "-c", program]) == 125
    receipt = json.loads((tmp_path / "receipts/resource-report.json").read_text())
    assert receipt["reason"] == "time_limit"
    assert receipt["cleanup_complete"]
    assert receipt["observed_processes"] >= 3


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_fast_orphan_is_owned_and_rejected(tmp_path: Path) -> None:
    program = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)"
    observer = StageObserver(tmp_path / "receipts", tmp_path, timeout=5, interval=0.05)
    assert observer.run([sys.executable, "-c", program]) == 125
    receipt = json.loads((tmp_path / "receipts/resource-report.json").read_text())
    assert receipt["reason"] == "descendant_leak"
    assert receipt["cleanup_complete"]


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_sampler_loss_stops_work_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer = StageObserver(tmp_path / "receipts", tmp_path, timeout=5, interval=0.05)
    monkeypatch.setattr(
        observer, "sample", lambda *args: (_ for _ in ()).throw(OSError("sampler lost"))
    )
    assert observer.run([sys.executable, "-c", "import time;time.sleep(30)"]) == 125
    receipt = json.loads((tmp_path / "receipts/resource-report.json").read_text())
    assert receipt["reason"] == "monitor_failed_or_interrupted"
    assert receipt["cleanup_complete"]
