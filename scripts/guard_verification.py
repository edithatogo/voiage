"""Bound a local verification process group without touching unrelated jobs.

Run with the repository's Python 3.15.1 interpreter. RSS is sampled, so this
guard limits runaway growth but is not a kernel-enforced allocation limit.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


def group_rss_kib(group: int) -> int:
    """Sum resident memory across the isolated job and all its group members."""
    result = subprocess.run(
        ["/bin/ps", "-axo", "pgid=,rss="], check=True, capture_output=True, text=True
    )
    return sum(
        int(fields[1])
        for line in result.stdout.splitlines()
        if len(fields := line.split()) == 2 and int(fields[0]) == group
    )


def stop_group(group: int) -> None:
    """Allow cleanup, then kill remaining children even if the leader exited."""
    for sig, delay in ((signal.SIGTERM, 3), (signal.SIGKILL, 0)):
        # macOS can report EPERM when only unreaped zero-RSS zombies remain.
        if group_rss_kib(group) == 0:
            return
        result = subprocess.run(  # noqa: S603 - fixed executable, typed group ID
            ["/bin/kill", f"-{sig.name.removeprefix('SIG')}", "--", str(-group)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode and "No such process" in result.stderr:
            return
        if result.returncode:
            raise RuntimeError(f"Cannot stop verification group: {result.stderr}")
        if delay:
            time.sleep(delay)


def clear_temporary_directory(path: Path) -> None:
    """Remove only this job's private tree, including read-only test fixtures."""
    for directory, _children, _files in os.walk(path, followlinks=False):
        Path(directory).chmod(0o700)
    shutil.rmtree(path)


def main() -> int:
    """Launch one job with memory, elapsed-time and free-space limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    parser.add_argument("--interval-seconds", type=float, default=2)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    def interrupted(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not command
        or min(args.max_rss_mib, args.timeout_seconds, args.interval_seconds) <= 0
        or args.min_free_mib < 0
    ):
        parser.error("Provide a command and positive resource limits")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = {"peak_rss_kib": 0, "reason": "starting", "returncode": None}
    child = None
    temp_dir = None
    started = time.monotonic()
    try:
        if shutil.disk_usage(args.work_dir).free < args.min_free_mib * 1024**2:
            report["reason"] = "insufficient_free_space"
            return 125
        # Build/consumer fixtures require temporary trees outside the checkout.
        # A unique sibling stays on the repository volume without nesting in it.
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=".voiage-verification-", dir=Path(__file__).resolve().parents[2]
            )
        )
        env = dict(os.environ, TMPDIR=str(temp_dir.resolve()))
        env.update(
            dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "POLARS_MAX_THREADS",
                ),
                "1",
            )
        )
        # Explicit operator-supplied argv; no shell interpolation.
        child = subprocess.Popen(command, env=env, start_new_session=True)  # noqa: S603
        report["reason"] = "completed"
        while child.poll() is None:
            rss = group_rss_kib(child.pid)
            report["peak_rss_kib"] = max(report["peak_rss_kib"], rss)
            if rss > args.max_rss_mib * 1024:
                report["reason"] = "memory_limit"
            elif time.monotonic() - started > args.timeout_seconds:
                report["reason"] = "time_limit"
            elif (
                min(
                    shutil.disk_usage(args.work_dir).free,
                    shutil.disk_usage(temp_dir).free,
                )
                < args.min_free_mib * 1024**2
            ):
                report["reason"] = "free_space_limit"
            if report["reason"] != "completed":
                stop_group(child.pid)
                child.wait()
                return 125
            time.sleep(args.interval_seconds)
        report["returncode"] = child.returncode
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    except BaseException:
        report["reason"] = "interrupted_or_monitor_failed"
        raise
    finally:
        if child is not None:
            stop_group(child.pid)
            child.wait()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["temporary_directory"] = str(temp_dir) if temp_dir else None
        try:
            if temp_dir is not None and report["returncode"] == 0:
                clear_temporary_directory(temp_dir)
        finally:
            (args.work_dir / "resource-report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
