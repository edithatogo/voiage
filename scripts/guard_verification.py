"""Bound a local verification process group without touching unrelated jobs.

Run with the repository's Python 3.15.1 interpreter. RSS is sampled, so this
guard limits runaway growth but is not a kernel-enforced allocation limit.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


class DiskMeasurements:
    """Sample each selected filesystem once; deltas are not job attribution."""

    def __init__(self, paths: list[Path]) -> None:
        self.records: dict[int, dict[str, Any]] = {}
        for selected in paths:
            path = selected.resolve(strict=True)
            device = path.stat().st_dev
            record = self.records.setdefault(
                device, {"device": device, "paths": [], "samples": 0}
            )
            if str(path) not in record["paths"]:
                record["paths"].append(str(path))

    def sample(self) -> int:
        """Record free bytes and return the lowest free space across devices."""
        for device, record in self.records.items():
            path = Path(record["paths"][0])
            if path.stat().st_dev != device:
                raise OSError(f"Filesystem changed during verification: {path}")
            usage = shutil.disk_usage(path)
            if not record["samples"]:
                record.update(start_free_bytes=usage.free, min_free_bytes=usage.free)
            record.update(
                total_bytes=usage.total,
                end_free_bytes=usage.free,
                min_free_bytes=min(record["min_free_bytes"], usage.free),
            )
            record["samples"] += 1
        return min(item["end_free_bytes"] for item in self.records.values())


def artifact_size(path: Path) -> dict[str, Any]:
    """Measure a selected tree once, without following links or mounted trees.

    Allocation uses st_blocks and deduplicates hardlinks within this snapshot.
    Reflinks, compression and shared blocks prevent physical-exclusive claims.
    """
    path = path.absolute()
    result: dict[str, Any] = {"path": str(path), "status": "absent"}
    try:
        root = path.lstat()
    except FileNotFoundError:
        return result
    if not stat.S_ISDIR(root.st_mode):
        raise ValueError(f"Artifact root must be a real directory: {path}")
    result.update(
        status="measured",
        regular_file_bytes=0,
        allocated_bytes=0,
        symlinks_skipped=0,
        cross_device_skipped=0,
    )
    seen: set[tuple[int, int]] = set()

    def visit(entry: Path) -> bool:
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode):
            result["symlinks_skipped"] += 1
            return False
        if info.st_dev != root.st_dev:
            result["cross_device_skipped"] += 1
            return False
        key = (info.st_dev, info.st_ino)
        if key not in seen:
            seen.add(key)
            result["allocated_bytes"] += info.st_blocks * 512
            if stat.S_ISREG(info.st_mode):
                result["regular_file_bytes"] += info.st_size
        return stat.S_ISDIR(info.st_mode)

    visit(path)

    def walk_error(error: OSError) -> None:
        raise error

    for directory, children, files in os.walk(
        path, followlinks=False, onerror=walk_error
    ):
        children[:] = [name for name in children if visit(Path(directory) / name)]
        for name in files:
            visit(Path(directory) / name)
    return result


def workload_identity(args: argparse.Namespace, command: list[str]) -> dict[str, Any]:
    """Bind the checked-out source, runtime, configuration and declared cache state."""
    root = (
        args.source_root.resolve(strict=True)
        if args.source_root is not None
        else Path(__file__).resolve().parents[1]
    )

    def git(*arguments: str) -> str:
        return subprocess.check_output(  # noqa: S603 - fixed read-only git arguments
            ["/usr/bin/git", "-C", str(root), *arguments], text=True
        ).strip()

    identity = {
        "id": args.workload_id,
        "cache_state": args.cache_state,
        "command": command,
        "python": sys.version,
        "executable": sys.executable,
        "source_root": str(root),
        "source_head": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--no-ext-diff").encode()
        ).hexdigest(),
        "untracked_paths": git(
            "ls-files", "--others", "--exclude-standard"
        ).splitlines(),
        "configuration_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("uv.lock", "pyproject.toml", "tox.ini")
        },
        "declared_configuration": {},
    }
    if args.identity_file:
        extra = json.loads(args.identity_file.read_text())
        if not isinstance(extra, dict):
            raise ValueError("Identity file must contain a JSON object")
        identity["declared_configuration"] = extra
    return identity


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
    parser.add_argument("--workload-id", default="unspecified")
    parser.add_argument(
        "--cache-state", choices=("cold", "warm", "mixed", "unknown"), default="unknown"
    )
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Repository root whose Git state and configuration identify the workload",
    )
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--disk-path", type=Path, action="append", default=[])
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
    report: dict[str, Any] = {
        "peak_rss_kib": 0,
        "reason": "starting",
        "returncode": None,
    }
    disk: DiskMeasurements | None = None
    report["disk"] = {
        "sampling_interval_seconds": args.interval_seconds,
        "filesystems": [],
        "artifacts_before": [],
        "artifacts_after": [],
        "physical_capacity_qualified": False,
        "limitations": [
            "Sampled free space can miss short peaks and includes unrelated activity.",
            "Directory snapshots are not peaks; allocated blocks may be shared.",
            "VM capacity requires separate guest and host-backing evidence.",
            "Cache state and additional configuration are operator declarations.",
        ],
    }
    child = None
    temp_dir = None
    started = time.monotonic()
    try:
        report["workload"] = workload_identity(args, command)
        # Check the temporary-file volume before creating or launching anything.
        disk = DiskMeasurements(
            [args.work_dir, Path(__file__).resolve().parents[2], *args.disk_path]
        )
        report["disk"]["filesystems"] = list(disk.records.values())
        if disk.sample() < args.min_free_mib * 1024**2:
            report["reason"] = "insufficient_free_space"
            return 125
        report["disk"]["artifacts_before"] = [
            artifact_size(p) for p in args.artifact_dir
        ]
        # Build/consumer fixtures require temporary trees outside the checkout.
        # A unique sibling stays on the repository volume without nesting in it.
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=".voiage-verification-", dir=Path(__file__).resolve().parents[2]
            )
        )
        env = dict(os.environ, TMPDIR=str(temp_dir.resolve()))
        # Test clones must not leave detached Git fsmonitor daemons behind.
        # Append a process-local override, preserving all existing Git settings.
        git_config_count = int(env.get("GIT_CONFIG_COUNT", "0"))
        env[f"GIT_CONFIG_KEY_{git_config_count}"] = "core.fsmonitor"
        env[f"GIT_CONFIG_VALUE_{git_config_count}"] = "false"
        env["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
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
            free = disk.sample()
            rss = group_rss_kib(child.pid)
            report["peak_rss_kib"] = max(report["peak_rss_kib"], rss)
            if rss > args.max_rss_mib * 1024:
                report["reason"] = "memory_limit"
            elif time.monotonic() - started > args.timeout_seconds:
                report["reason"] = "time_limit"
            elif free < args.min_free_mib * 1024**2:
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
        try:
            if child is not None:
                stop_group(child.pid)
                child.wait()
            if disk is not None:
                disk.sample()
            report["disk"]["artifacts_after"] = [
                artifact_size(p) for p in args.artifact_dir
            ]
            if temp_dir is not None and report["returncode"] == 0:
                clear_temporary_directory(temp_dir)
        except BaseException:
            report["reason"] = "interrupted_or_monitor_failed"
            raise
        finally:
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            report["temporary_directory"] = str(temp_dir) if temp_dir else None
            (args.work_dir / "resource-report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
