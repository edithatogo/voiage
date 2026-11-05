"""Bound one Linux compiler stage and retain terminal resource evidence."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import psutil

from scripts.guard_verification import artifact_size
from scripts.observe_arrow_capacity import ObservedProcesses, descendant_usage


class StageObserver:
    """Observe a single owned stage, including orphaned and detached children."""

    def __init__(
        self,
        evidence: Path,
        workspace: Path,
        *,
        timeout: float = 18000,
        interval: float = 2,
        max_rss_mib: int = 12288,
        min_free_mib: int = 8192,
    ) -> None:
        self.evidence = evidence
        self.workspace = workspace
        self.timeout = timeout
        self.interval = interval
        self.max_rss_mib = max_rss_mib
        self.min_free_mib = min_free_mib
        self.usage: dict[tuple[int, float], dict[str, float]] = {}
        self.report: dict[str, Any] = {}

    def sample(self, samples: Any) -> None:
        """Sample owned descendants and cumulative counters without PID reuse."""
        pids, tree_rss = descendant_usage(os.getpid())
        pids.discard(os.getpid())
        self.tracker.observe(pids, os.getpid())
        rss = max(0, tree_rss * 1024 - psutil.Process().memory_info().rss)
        for pid in pids:
            try:
                process = psutil.Process(pid)
                key = pid, process.create_time()
                cpu = process.cpu_times()
                counters = self.usage.setdefault(
                    key,
                    {
                        "user_seconds": 0,
                        "system_seconds": 0,
                        "read_bytes": 0,
                        "write_bytes": 0,
                    },
                )
                counters.update(user_seconds=cpu.user, system_seconds=cpu.system)
                try:
                    io = process.io_counters()
                    counters.update(
                        read_bytes=io.read_bytes, write_bytes=io.write_bytes
                    )
                except psutil.AccessDenied:
                    self.report.setdefault("io_counter_gaps", []).append(
                        {
                            "pid": pid,
                            "reason": "permission_denied_during_exec_or_runtime",
                        }
                    )
            except psutil.NoSuchProcess, psutil.ZombieProcess:
                continue
        free = psutil.disk_usage(self.workspace).free
        self.report["peak_rss_kib"] = max(self.report["peak_rss_kib"], rss // 1024)
        self.report["minimum_free_bytes"] = min(self.report["minimum_free_bytes"], free)
        row = {
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "rss_bytes": rss,
            "free_bytes": free,
            "live_descendants": len(pids),
            "sampled_totals": {
                k: sum(v[k] for v in self.usage.values())
                for k in ("user_seconds", "system_seconds", "read_bytes", "write_bytes")
            },
        }
        samples.write(json.dumps(row) + "\n")
        samples.flush()
        if rss > self.max_rss_mib * 1024**2:
            self.report["reason"] = "memory_limit"
        elif free < self.min_free_mib * 1024**2:
            self.report["reason"] = "free_space_limit"
        elif time.monotonic() - self.started > self.timeout:
            self.report["reason"] = "time_limit"

    def run(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> int:
        """Run with GNU time and Linux subreaper ownership; fail on lost monitoring."""
        if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
            raise RuntimeError("Linux subreaper and pidfd support are required")
        if psutil.Process().children():
            raise RuntimeError("Observer must start without existing children")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "Cannot establish orphan ownership")
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.tracker = ObservedProcesses()
        self.report = {
            "command": command,
            "cwd": str(cwd or Path.cwd()),
            "reason": "completed",
            "returncode": None,
            "peak_rss_kib": 0,
            "minimum_free_bytes": psutil.disk_usage(self.workspace).free,
            "limits": {
                "rss_mib": self.max_rss_mib,
                "timeout_seconds": self.timeout,
                "min_free_mib": self.min_free_mib,
            },
            "interval_seconds": self.interval,
            "gnu_time_available": Path("/usr/bin/time").is_file(),
            "cleanup_complete": False,
            "subreaper": True,
            "python": sys.version,
            "limitations": [
                "RSS sums shared pages and can miss short-lived peaks.",
                "CPU and IO samples can miss short-lived processes; GNU time retains wait4 totals.",
                "Filesystem free space includes unrelated activity; directory snapshots are not peaks.",
            ],
        }
        process = None
        result = 125
        previous = signal.getsignal(signal.SIGTERM)

        def interrupt(_signum: int, _frame: object) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        try:
            if not self.report["gnu_time_available"]:
                raise RuntimeError("GNU time is required")  # noqa: TRY301 - terminal receipt is mandatory
            with (self.evidence / "samples.jsonl").open("w") as samples:
                self.sample(samples)
                if self.report["reason"] != "completed":
                    return 125
                with (self.evidence / "command.log").open("wb") as output:
                    process = subprocess.Popen(  # noqa: S603 - explicit trusted stage argv, no shell
                        [
                            "/usr/bin/time",
                            "-v",
                            "-o",
                            str(self.evidence / "gnu-time.txt"),
                            "--",
                            *command,
                        ],
                        cwd=cwd,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    while process.poll() is None:
                        self.sample(samples)
                        if self.report["reason"] != "completed":
                            break
                        time.sleep(self.interval)
                    if self.report["reason"] == "completed":
                        self.report["returncode"] = process.wait()
                        self.sample(samples)
                        children = [
                            p
                            for p in psutil.Process().children(recursive=True)
                            if p.status() != psutil.STATUS_ZOMBIE
                        ]
                        if children:
                            self.report["reason"] = "descendant_leak"
                        elif process.returncode:
                            self.report["reason"] = "command_failed"
                        result = (
                            process.returncode
                            if process.returncode >= 0
                            else 128 - process.returncode
                        )
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            self.report.update(
                reason="monitor_failed_or_interrupted", error=repr(error)
            )
        finally:
            try:
                self.report["trigger_reason"] = self.report["reason"]
                # A controller can fork between the final sample and its stop.
                # The subreaper owns those newly orphaned identities too.
                for attempt in range(3):
                    owned, _rss = descendant_usage(os.getpid())
                    owned.discard(os.getpid())
                    self.tracker.observe(owned, os.getpid())
                    self.tracker.stop()
                    if process is not None:
                        process.wait(timeout=10)
                    while True:
                        try:
                            pid, _status = os.waitpid(-1, os.WNOHANG)
                        except ChildProcessError:
                            break
                        if pid == 0:
                            break
                    self.report["cleanup_passes"] = attempt + 1
                    self.report["cleanup_complete"] = not psutil.Process().children()
                    if self.report["cleanup_complete"]:
                        break
                if not self.report["cleanup_complete"]:
                    raise RuntimeError("Owned descendants remain")  # noqa: TRY301 - cleanup failure must be recorded
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                self.report.update(reason="cleanup_failed", cleanup_error=repr(error))
            finally:
                self.tracker.close()
                signal.signal(signal.SIGTERM, previous)
                self.report["observed_processes"] = len(self.usage)
                self.report["sampled_totals"] = {
                    k: sum(v[k] for v in self.usage.values())
                    for k in (
                        "user_seconds",
                        "system_seconds",
                        "read_bytes",
                        "write_bytes",
                    )
                }
                self.report["elapsed_seconds"] = round(
                    time.monotonic() - self.started, 3
                )
                self.report["artifacts_after"] = artifact_size(self.workspace)
                (self.evidence / "resource-report.json").write_text(
                    json.dumps(self.report, indent=2) + "\n"
                )
        return (
            result if self.report["reason"] in {"completed", "command_failed"} else 125
        )


def main() -> int:
    """Run a trusted stage under a root-owned monitor and unprivileged net namespace."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.geteuid() != 0 or args.uid <= 0 or args.gid <= 0:
        parser.error("Root monitor and non-root build UID/GID are required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A trusted stage command is required")
    isolated = [
        "/usr/bin/unshare",
        "--net",
        "--",
        sys.executable,
        "-m",
        "scripts.enter_native_namespace",
        "--parent-namespace",
        os.readlink("/proc/self/ns/net"),
        "--uid",
        str(args.uid),
        "--gid",
        str(args.gid),
        "--",
        *command,
    ]
    observer = StageObserver(args.evidence, args.workspace, timeout=args.timeout)
    return observer.run(isolated, cwd=args.cwd)


if __name__ == "__main__":
    raise SystemExit(main())
