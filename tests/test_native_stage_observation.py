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
    observer = StageObserver(
        tmp_path / "receipts", tmp_path, timeout=5, interval=0.05, min_free_mib=1
    )
    assert observer.run([sys.executable, "-c", "raise SystemExit(7)"]) == 7, (
        observer.report
    )
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
        tmp_path / "receipts", tmp_path, timeout=0.4, interval=0.05, min_free_mib=1
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
    observer = StageObserver(
        tmp_path / "receipts", tmp_path, timeout=5, interval=0.05, min_free_mib=1
    )
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
    observer = StageObserver(
        tmp_path / "receipts", tmp_path, timeout=5, interval=0.05, min_free_mib=1
    )
    monkeypatch.setattr(
        observer, "sample", lambda *args: (_ for _ in ()).throw(OSError("sampler lost"))
    )
    assert observer.run([sys.executable, "-c", "import time;time.sleep(30)"]) == 125
    receipt = json.loads((tmp_path / "receipts/resource-report.json").read_text())
    assert receipt["reason"] == "monitor_failed_or_interrupted"
    assert receipt["cleanup_complete"]


@pytest.mark.parametrize("io_capability", ["denied", "unsupported"])
def test_io_permission_gap_does_not_disable_the_rss_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, io_capability: str
) -> None:
    import io
    import time
    from types import SimpleNamespace

    import psutil

    observer = StageObserver(tmp_path / "receipts", tmp_path, max_rss_mib=1)
    observer.started = time.monotonic()
    observer.report = {
        "reason": "completed",
        "peak_rss_kib": 0,
        "minimum_free_bytes": 10**12,
    }
    observer.tracker = SimpleNamespace(observe=lambda *args: None)
    monkeypatch.setattr(
        "scripts.observe_native_stage.descendant_usage", lambda pid: ({pid, 42}, 4096)
    )

    class Process:
        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def create_time(self):
            return 1.0

        def cpu_times(self):
            return SimpleNamespace(user=1.0, system=0.1)

        def io_counters(self):
            raise psutil.AccessDenied(42)

    if io_capability == "unsupported":
        delattr(Process, "io_counters")
    monkeypatch.setattr(
        "scripts.observe_native_stage.psutil.Process", lambda *args: Process()
    )
    observer.sample(io.StringIO())
    assert observer.report["reason"] == "memory_limit"
    assert observer.report["io_counter_gaps"][0]["pid"] == 42
    assert observer.report["io_measurement_complete"] is False
    if io_capability == "unsupported":
        assert observer.report["io_counters_supported"] is False


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_sigterm_restores_handler_and_cleans_owned_children(tmp_path: Path) -> None:
    import os
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    observer = StageObserver(
        tmp_path / "receipts", tmp_path, timeout=5, interval=0.05, min_free_mib=1
    )
    command = [
        sys.executable,
        "-c",
        "import os,signal,sys,time;time.sleep(.15);os.kill(int(sys.argv[1]),signal.SIGTERM);time.sleep(30)",
        str(os.getpid()),
    ]
    assert observer.run(command) == 125
    assert observer.report["reason"] == "monitor_failed_or_interrupted"
    assert observer.report["cleanup_complete"]
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_prelaunch_capacity_stop_still_restores_handler(tmp_path: Path) -> None:
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    observer = StageObserver(tmp_path / "receipts", tmp_path, min_free_mib=10**12)
    assert (
        observer.run([sys.executable, "-c", "raise AssertionError('must not launch')"])
        == 125
    )
    assert observer.report["reason"] == "free_space_limit"
    assert observer.report["cleanup_complete"]
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux subreaper and pidfds are required"
)
def test_cleanup_adopts_a_child_created_after_the_last_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from scripts.observe_arrow_capacity import ObservedProcesses

    children = []

    class ForkDuringStop(ObservedProcesses):
        first = True

        def stop(self) -> None:
            if self.first:
                self.first = False
                children.append(
                    subprocess.Popen(
                        [sys.executable, "-c", "import time;time.sleep(30)"],
                        start_new_session=True,
                    )
                )
            super().stop()

    monkeypatch.setattr(
        "scripts.observe_native_stage.ObservedProcesses", ForkDuringStop
    )
    observer = StageObserver(
        tmp_path / "receipts", tmp_path, min_free_mib=1, timeout=5, interval=0.05
    )
    assert observer.run([sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert observer.report["cleanup_complete"]
    assert observer.report["cleanup_passes"] == 2
    assert observer.report["trigger_reason"] == "command_failed"
