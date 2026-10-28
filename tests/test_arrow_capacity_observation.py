"""Negative controls for source and native-provenance observation claims."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.observe_arrow_capacity import (
    Observation,
    ObservedProcesses,
    descendant_usage,
    download,
    native_objects,
    validate_linkage,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("exit_code", [0, 125])
def test_guard_exit_stops_observed_detached_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    observation = Observation(tmp_path / "workspace", tmp_path / "evidence", inputs)
    events: list[object] = []

    class Tracker:
        def observe(self, pids: set[int], _root: int) -> None:
            events.append(("observe", pids))

        def stop(self) -> None:
            events.append("stop")

        def close(self) -> None:
            events.append("close")

    class Process:
        pid = 100
        returncode = exit_code
        polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else exit_code

        def wait(self) -> int:
            return exit_code

    monkeypatch.setattr("scripts.observe_arrow_capacity.ObservedProcesses", Tracker)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.subprocess.Popen", lambda *a, **k: Process()
    )
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.descendant_usage", lambda _pid: ({100, 101}, 1)
    )
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.time.sleep", lambda _seconds: None
    )
    if exit_code:
        with pytest.raises(RuntimeError, match="exit 125"):
            observation.run("guard-exit", ["test"], {})
    else:
        observation.run("guard-exit", ["test"], {})
    assert events == [("observe", {100, 101}), "stop", "close"]


def test_tracker_prunes_finished_handles_and_terminates_retained_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    sent: list[tuple[int, int]] = []
    ready: set[int] = set()

    class Poller:
        def __init__(self) -> None:
            self.registered: list[int] = []

        def register(self, fd: int, _events: int) -> None:
            self.registered.append(fd)

        def unregister(self, fd: int) -> None:
            self.registered.remove(fd)

        def poll(self, timeout: int) -> list[tuple[int, int]]:
            return [(fd, 1) for fd in (ready if timeout == 0 else self.registered)]

    monkeypatch.setattr("scripts.observe_arrow_capacity.select.poll", Poller)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.descendant_usage",
        lambda _root: ({10, 11, 12}, 0),
    )
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.os.pidfd_open",
        lambda pid: pid + 1000,
        raising=False,
    )
    monkeypatch.setattr("scripts.observe_arrow_capacity.os.close", closed.append)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.signal.pidfd_send_signal",
        lambda fd, sig: sent.append((fd, sig)),
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.time.sleep", lambda _seconds: None
    )
    tracker = ObservedProcesses()
    tracker.observe({10, 11}, 10)
    ready.add(1010)
    tracker.observe({11, 12}, 10)
    assert closed == [1010]
    tracker.stop()
    tracker.close()
    assert set(sent) == {
        (1011, signal.SIGTERM),
        (1012, signal.SIGTERM),
        (1011, signal.SIGKILL),
        (1012, signal.SIGKILL),
    }
    assert set(closed) == {1010, 1011, 1012}


def test_tracker_reports_descendant_cleanup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Poller:
        def register(self, _fd: int, _events: int) -> None:
            pass

        def poll(self, _timeout: int) -> list[tuple[int, int]]:
            return []

    times = iter([0.0, 0.0, 6.0])
    monkeypatch.setattr("scripts.observe_arrow_capacity.select.poll", Poller)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.signal.pidfd_send_signal",
        lambda _fd, _sig: None,
        raising=False,
    )
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.time.sleep", lambda _seconds: None
    )
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.time.monotonic", lambda: next(times)
    )
    tracker = ObservedProcesses()
    tracker.handles[10] = 1010
    with pytest.raises(RuntimeError, match="did not terminate"):
        tracker.stop()


def test_tracker_rejects_pid_no_longer_in_descendant_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []

    class Poller:
        def register(self, _fd: int, _events: int) -> None:
            pass

        def poll(self, _timeout: int) -> list[tuple[int, int]]:
            return []

    monkeypatch.setattr("scripts.observe_arrow_capacity.select.poll", Poller)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.os.pidfd_open",
        lambda pid: pid + 1000,
        raising=False,
    )
    monkeypatch.setattr("scripts.observe_arrow_capacity.os.close", closed.append)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.descendant_usage", lambda _root: ({10}, 0)
    )
    tracker = ObservedProcesses()
    tracker.observe({10, 11}, 10)
    assert tracker.handles == {10: 1010}
    assert closed == [1011]
    tracker.close()


def test_tracker_closes_new_handles_if_revalidation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []

    class Poller:
        def register(self, _fd: int, _events: int) -> None:
            pass

        def poll(self, _timeout: int) -> list[tuple[int, int]]:
            return []

    def fail(_root: int) -> tuple[set[int], int]:
        raise RuntimeError("process sampling failed")

    monkeypatch.setattr("scripts.observe_arrow_capacity.select.poll", Poller)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.os.pidfd_open",
        lambda pid: pid + 1000,
        raising=False,
    )
    monkeypatch.setattr("scripts.observe_arrow_capacity.os.close", closed.append)
    monkeypatch.setattr("scripts.observe_arrow_capacity.descendant_usage", fail)
    tracker = ObservedProcesses()
    with pytest.raises(RuntimeError, match="process sampling failed"):
        tracker.observe({10, 11}, 10)
    assert set(closed) == {1010, 1011}
    assert tracker.handles == {}


def test_cleanup_failure_is_recorded_and_handles_are_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    observation = Observation(tmp_path / "workspace", tmp_path / "evidence", inputs)
    closed: list[bool] = []

    class Tracker:
        def stop(self) -> None:
            raise RuntimeError("descendant did not stop")

        def close(self) -> None:
            closed.append(True)

    class Process:
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr("scripts.observe_arrow_capacity.ObservedProcesses", Tracker)
    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.subprocess.Popen", lambda *a, **k: Process()
    )
    with pytest.raises(RuntimeError, match="descendant did not stop"):
        observation.run("cleanup-failure", ["test"], {})
    receipt = json.loads(
        (observation.evidence / "cleanup-failure-descendant-resources.json").read_text()
    )
    assert receipt["reason"] == "descendant_cleanup_failed"
    assert receipt["cleanup_error"] == "descendant did not stop"
    assert closed == [True]


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="Linux pidfd process handles are required",
)
def test_tracker_stops_observed_child_after_parent_exits() -> None:
    """A separate-session child stays bound to its original process handle."""
    program = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "start_new_session=True); "
        "print(p.pid,flush=True); "
        "signal=sys.stdin.read(1); "
        "p.terminate() if signal != 'q' else None; "
        "p.wait() if signal != 'q' else None"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    tracker = ObservedProcesses()
    try:
        assert parent.stdout is not None
        assert parent.stdin is not None
        child = int(parent.stdout.readline().strip())
        tracker.observe({parent.pid, child}, parent.pid)
        child_handle = tracker.handles[child]
        parent.stdin.write("q")
        parent.stdin.flush()
        assert parent.wait(timeout=10) == 0
        tracker.stop()
        poller = select.poll()
        poller.register(child_handle, select.POLLIN)
        assert poller.poll(10_000), "observed child survived guard termination"
    finally:
        if parent.stdin:
            parent.stdin.close()
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=10)
        tracker.stop()
        tracker.close()


def test_linkage_accepts_both_built_libraries(tmp_path: Path) -> None:
    prefix = tmp_path / "install"
    text = "\n".join(
        f"{name}.so.2600 => {prefix}/lib/{name}.so.2600 (0x1234)"
        for name in ("libarrow", "libparquet", "libarrow_dataset")
    )
    validate_linkage(text, prefix)


@pytest.mark.parametrize(
    "damage", ["outside-prefix", "unresolved", "empty", "missing-parquet"]
)
def test_linkage_rejects_false_native_acceptance(tmp_path: Path, damage: str) -> None:
    prefix = tmp_path / "install"
    text = (
        f"libarrow.so.2600 => {prefix}/lib/libarrow.so.2600 (0x1234)\n"
        f"libparquet.so.2600 => {prefix}/lib/libparquet.so.2600 (0x1234)"
    )
    if damage == "outside-prefix":
        text = text.replace(
            str(prefix) + "/lib/libarrow", str(tmp_path / "other/lib/libarrow")
        )
    elif damage == "unresolved":
        text += "\nlibarrow_compute.so.2600 => not found"
    elif damage == "empty":
        text = ""
    elif damage == "missing-parquet":
        text = text.splitlines()[0]
    with pytest.raises(RuntimeError, match="Native linkage"):
        validate_linkage(text, prefix)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "https://example.com/source.tar.gz"]
)
def test_source_download_rejects_nonprovider_sources(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported source host"):
        download({"url": url}, tmp_path)


def test_native_object_scan_includes_parquet_extensions(tmp_path: Path) -> None:
    names = ["lib.cpython-315.so", "_parquet.cpython-315.so", "libarrow_python.so.2600"]
    for name in names:
        (tmp_path / name).write_bytes(b"\x7fELF")
    (tmp_path / "metadata.txt").touch()
    (tmp_path / "v0.7.1.some-named-index.parquet").write_bytes(b"PAR1")
    (tmp_path / "not-elf.so").write_text("linker script")
    assert {path.name for path in native_objects(tmp_path, tmp_path)} == set(names)


def test_native_object_scan_rejects_another_environment(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="escaped"):
        native_objects(tmp_path.parent, tmp_path)


def test_resource_sampling_counts_children_in_other_process_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Table:
        stdout = "100 50 100 20\n101 100 101 100\n102 101 102 10000\n200 50 200 99999\n"

    monkeypatch.setattr(
        "scripts.observe_arrow_capacity.subprocess.run",
        lambda *_args, **_kwargs: Table(),
    )
    pids, rss = descendant_usage(100)
    assert pids == {100, 101, 102}
    assert rss == 10120


def test_sampler_reports_unexpected_base_exception_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    observation = Observation(tmp_path / "workspace", tmp_path / "evidence", inputs)

    def fail(_path: Path) -> None:
        raise SystemExit("sampling interrupted")

    monkeypatch.setattr("scripts.observe_arrow_capacity.shutil.disk_usage", fail)
    with pytest.raises(SystemExit, match="sampling interrupted"):
        observation.sample()
    assert observation.monitor_errors
