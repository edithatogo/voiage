"""Measure an isolated Arrow source build without asserting HPC acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any
import urllib.request


def descendant_usage(root_pid: int) -> tuple[set[int], int]:
    """Count the process tree even when Ninja gives children new process groups."""
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        tuple(map(int, line.split()))
        for line in result.stdout.splitlines()
        if len(line.split()) == 4
    ]
    descendants = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for pid, ppid, _pgid, _rss in rows:
            if ppid == parent and pid not in descendants:
                descendants.add(pid)
                pending.append(pid)
    rss = sum(size for pid, _ppid, _pgid, size in rows if pid in descendants)
    return descendants, rss


class ObservedProcesses:
    """Retain live Linux process handles without accumulating exited children."""

    def __init__(self) -> None:
        self.handles: dict[int, int] = {}

    def observe(self, pids: set[int], root_pid: int) -> None:
        """Retain handles while descendants are still attached to the guard."""
        poller = select.poll()
        for handle in self.handles.values():
            poller.register(handle, select.POLLIN)
        finished = {handle for handle, _ in poller.poll(0)}
        for pid, handle in list(self.handles.items()):
            if handle in finished:
                os.close(handle)
                del self.handles[pid]
        opened: dict[int, int] = {}
        try:
            for pid in pids - self.handles.keys():
                try:
                    opened[pid] = os.pidfd_open(pid)  # ty: ignore[unresolved-attribute] - Linux-only workload
                except ProcessLookupError:
                    continue
            if opened:
                verified, _ = descendant_usage(root_pid)
                for pid in opened.keys() & verified:
                    self.handles[pid] = opened.pop(pid)
        finally:
            for handle in opened.values():
                os.close(handle)

    def stop(self) -> None:
        """Stop only retained process identities, including detached groups."""
        if not self.handles:
            return
        for signum in (signal.SIGTERM, signal.SIGKILL):
            signalled = False
            for handle in self.handles.values():
                try:
                    signal.pidfd_send_signal(handle, signum)  # ty: ignore[unresolved-attribute] - Linux-only workload
                    signalled = True
                except ProcessLookupError:
                    continue
            if signum == signal.SIGTERM and signalled:
                time.sleep(3)
        poller = select.poll()
        remaining = set(self.handles.values())
        for handle in remaining:
            poller.register(handle, select.POLLIN)
        deadline = time.monotonic() + 5
        while remaining:
            timeout = max(0, int((deadline - time.monotonic()) * 1000))
            for handle, _ in poller.poll(timeout):
                poller.unregister(handle)
                remaining.discard(handle)
            if time.monotonic() >= deadline:
                break
        if remaining:
            raise RuntimeError(
                "Observed descendants did not terminate within five seconds"
            )

    def close(self) -> None:
        """Release every retained kernel handle."""
        for handle in self.handles.values():
            os.close(handle)
        self.handles.clear()


def native_objects(package: Path, environment: Path) -> list[Path]:
    """Include underscore-prefixed Python extensions as well as helper libraries."""
    if not package.resolve().is_relative_to(environment.resolve()):
        raise RuntimeError("Installed extension escaped the private environment")
    objects = []
    for path in package.rglob("*.so*"):
        if not path.is_file() or re.search(r"\.so(?:\.\d+)*$", path.name) is None:
            continue
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                objects.append(path)
    objects.sort()
    if not objects:
        raise RuntimeError("No installed PyArrow native objects found")
    return objects


class Observation:
    """Retain source-build stages and sampled filesystem use, including failures."""

    def __init__(self, workspace: Path, evidence: Path, inputs: Path) -> None:
        self.workspace = workspace
        self.evidence = evidence
        workspace.mkdir()
        evidence.mkdir()
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.monitor_errors: list[str] = []
        self.initial_free = shutil.disk_usage(workspace).free
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Git SDK is required")
        self.receipt: dict[str, Any] = {
            "scope": "Arrow/PyArrow source capacity; not Spack or full HPC acceptance",
            "commit": subprocess.check_output(  # noqa: S603 - qualified Git SDK with fixed arguments
                [git, "rev-parse", "HEAD"], text=True
            ).strip(),
            "run_url": os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            + "/"
            + os.environ.get("GITHUB_REPOSITORY", "local")
            + "/actions/runs/"
            + os.environ.get("GITHUB_RUN_ID", "local"),
            "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "top_level_build_jobs": 2,
            "bundled_dependency_parallelism": "Upstream processor-count defaults; nested commands may exceed the top-level limit",
            "initial_free_bytes": self.initial_free,
            "minimum_free_bytes": self.initial_free,
            "maximum_sampled_disk_growth_bytes": 0,
            "sampling_interval_seconds": 5,
            "steps": [],
            "outcome": "running",
            "full_hpc_matrix_qualified": False,
            "compiler_source_build_executed": False,
        }

    def save_locked(self) -> None:
        """Atomically write a snapshot while the caller holds the receipt lock."""
        pending = self.evidence / "receipt.pending"
        pending.write_text(json.dumps(self.receipt, indent=2) + "\n")
        pending.replace(self.evidence / "receipt.json")

    def sample(self) -> None:
        """Propagate sampler errors instead of accepting incomplete observations."""
        completed = False
        try:
            with (self.evidence / "filesystem-samples.jsonl").open("a") as output:
                while True:
                    usage = shutil.disk_usage(self.workspace)
                    with self.lock:
                        self.receipt["minimum_free_bytes"] = min(
                            self.receipt["minimum_free_bytes"], usage.free
                        )
                        self.receipt["maximum_sampled_disk_growth_bytes"] = max(
                            self.receipt["maximum_sampled_disk_growth_bytes"],
                            self.initial_free - usage.free,
                        )
                        output.write(
                            json.dumps(
                                {
                                    "elapsed_seconds": round(
                                        time.monotonic() - self.started, 2
                                    ),
                                    "free_bytes": usage.free,
                                    "used_bytes": usage.used,
                                }
                            )
                            + "\n"
                        )
                        output.flush()
                        self.save_locked()
                    if self.stop.wait(5):
                        break
            completed = True
        except Exception as exc:
            self.monitor_errors.append(repr(exc))
        finally:
            if not completed and not self.monitor_errors:
                self.monitor_errors.append("sampler exited unexpectedly")

    def run(
        self, name: str, command: list[str], env: dict[str, str], seconds: int = 1800
    ) -> None:
        """Bound each actual subprocess group and preserve its exit and resource report."""
        guard = [
            sys.executable,
            "scripts/guard_verification.py",
            "--max-rss-mib",
            "11264",
            "--min-free-mib",
            "2048",
            "--timeout-seconds",
            str(seconds),
            "--work-dir",
            str(self.evidence / (name + "-guard")),
            "--",
        ]
        row = {"name": name, "argv": command, "exit_code": None}
        with self.lock:
            self.receipt["steps"].append(row)
            self.save_locked()
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        resources: dict[str, Any] = {
            "scope": "Sampled descendants, including separate process groups; short-lived processes may be missed",
            "peak_rss_kib": 0,
            "sampling_interval_seconds": 2,
            "reason": "completed",
            "returncode": None,
        }
        with (self.evidence / (name + ".log")).open("w") as output:
            process = subprocess.Popen(  # noqa: S603 - fixed tools, structured source-build arguments
                guard + command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            observed = ObservedProcesses()
            try:
                while process.poll() is None:
                    owned, rss = descendant_usage(process.pid)
                    observed.observe(owned, process.pid)
                    resources["peak_rss_kib"] = max(resources["peak_rss_kib"], rss)
                    if rss > 11264 * 1024:
                        resources["reason"] = "descendant_memory_limit"
                        observed.stop()
                        break
                    time.sleep(2)
                resources["returncode"] = process.wait()
            except BaseException:
                resources["reason"] = "interrupted_or_monitor_failed"
                raise
            finally:
                try:
                    observed.stop()
                except Exception as error:
                    resources["reason"] = "descendant_cleanup_failed"
                    resources["cleanup_error"] = str(error)
                    raise
                finally:
                    observed.close()
                    resources["elapsed_seconds"] = round(time.monotonic() - started, 2)
                    (self.evidence / (name + "-descendant-resources.json")).write_text(
                        json.dumps(resources, indent=2) + "\n"
                    )
        with self.lock:
            row.update(
                exit_code=process.returncode,
                descendant_resource_reason=resources["reason"],
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
            self.save_locked()
        print(f"Completed {name}: exit {process.returncode}", flush=True)
        if process.returncode or resources["reason"] != "completed":
            raise RuntimeError(
                f"{name}: exit {process.returncode}, {resources['reason']}; see retained resource receipts"
            )


def download(source: dict[str, Any], workspace: Path) -> Path:
    """Download a bounded, checksum-bound upstream archive without extracting it."""
    url = source["url"]
    if not url.startswith(
        ("https://github.com/", "https://files.pythonhosted.org/packages/")
    ):
        raise ValueError("Unsupported source host")
    filename = source.get("filename", "mimalloc-v3.4.1.tar.gz")
    if Path(filename).name != filename:
        raise ValueError("Source filename must be a basename")
    target = workspace / filename
    with (
        urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310 - HTTPS source hosts checked above
        target.open("xb") as output,
    ):
        size = 0
        while chunk := response.read(65536):
            size += len(chunk)
            if size > 64 * 1024**2:
                raise ValueError("Source archive exceeds 64 MiB experiment budget")
            output.write(chunk)
    with target.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != source["sha256"]:
        raise ValueError(f"Source checksum mismatch: {filename}")
    return target


def unpack(archive: Path, destination: Path) -> Path:
    """Use Python 3.15's data filter and require exactly one upstream root."""
    destination.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected one upstream source root")
    return roots[0]


def validate_linkage(text: str, prefix: Path) -> None:
    """Reject missing core libraries or linkage escaping the exact build prefix."""
    targets = re.findall(
        r"(lib(?:arrow(?:_compute|_dataset|_acero)?|parquet))\.so[^\s]* => (/[^\s]+)",
        text,
    )
    names = {name for name, _path in targets}
    if (
        "not found" in text
        or not {"libarrow", "libparquet"}.issubset(names)
        or any(
            not Path(path).resolve().is_relative_to(prefix.resolve())
            for _name, path in targets
        )
    ):
        raise RuntimeError(
            "Native linkage did not resolve the built Arrow libraries inside the experiment prefix"
        )


def main() -> int:
    """Compile the frozen Arrow/PyArrow profile with its installed native probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1):
        raise RuntimeError("Select Python 3.15.1 explicitly")
    if sys.platform != "linux":
        raise RuntimeError("This experiment requires an isolated Linux builder")
    args.workspace = args.workspace.resolve()
    args.evidence = args.evidence.resolve()
    packet = json.loads(args.inputs.read_text())
    if packet["python_profile"] != "3.15.1":
        raise ValueError("Unexpected Python input profile")
    arrow, frontend = (packet["providers"][key] for key in ("arrow", "py-pyarrow"))
    profile = frontend["native_profile"]
    expected = {
        "brotli",
        "bz2",
        "compute",
        "csv",
        "dataset",
        "filesystem",
        "ipc",
        "lz4",
        "mimalloc",
        "parquet",
        "python",
        "shared",
        "snappy",
        "zlib",
        "zstd",
    }
    if set(profile.split("+")[1:]) != expected or not profile.startswith(
        "arrow@=26.0.0+"
    ):
        raise ValueError("The complete frozen Arrow I/O/codec profile is required")
    observation = Observation(args.workspace, args.evidence, args.inputs)
    shutil.copy2(args.inputs, args.evidence / "inputs.json")
    env = dict(
        os.environ,
        UV_CACHE_DIR=str(args.workspace / "uv-cache"),
        CMAKE_BUILD_PARALLEL_LEVEL="2",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    monitor = threading.Thread(target=observation.sample)
    monitor.start()
    try:
        cpp_archive = download(arrow["source"], args.workspace)
        python_archive = download(frontend["source"], args.workspace)
        allocator = download(arrow["mimalloc"], args.workspace)
        cpp = unpack(cpp_archive, args.workspace / "cpp-source")
        unpack(python_archive, args.workspace / "python-source")
        env["ARROW_MIMALLOC_URL"] = allocator.as_uri()
        prefix = args.workspace / "install"
        build = args.workspace / "cpp-build"
        venv = args.workspace / "venv"
        observation.run("compiler-sdk", ["g++", "--version"], env, 60)
        observation.run("cmake-sdk", ["cmake", "--version"], env, 60)
        observation.run(
            "create-environment",
            ["uv", "venv", "--python", sys.executable, str(venv)],
            env,
            60,
        )
        python = str(venv / "bin/python")
        observation.run(
            "numpy-sdk",
            ["uv", "pip", "install", "--python", python, "numpy==2.5.4"],
            env,
            300,
        )
        flags = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DPython3_EXECUTABLE={python}",
            "-DARROW_DEPENDENCY_SOURCE=BUNDLED",
            "-DARROW_BUILD_SHARED=ON",
            "-DARROW_BUILD_STATIC=OFF",
            "-DARROW_BUILD_TESTS=OFF",
            "-DARROW_BUILD_EXAMPLES=OFF",
            "-DARROW_NO_DEPRECATED_API=ON",
        ]
        features = [
            "COMPUTE",
            "CSV",
            "DATASET",
            "FILESYSTEM",
            "IPC",
            "MIMALLOC",
            "PARQUET",
            "PYTHON",
        ]
        flags += ["-DARROW_" + feature + "=ON" for feature in features]
        flags += [
            "-DARROW_WITH_" + codec + "=ON"
            for codec in ("BROTLI", "BZ2", "LZ4", "SNAPPY", "ZLIB", "ZSTD")
        ]
        observation.run(
            "cpp-configure",
            ["cmake", "-S", str(cpp / "cpp"), "-B", str(build), "-G", "Ninja", *flags],
            env,
            900,
        )
        shutil.copy2(build / "CMakeCache.txt", args.evidence / "CMakeCache.txt")
        shutil.copy2(build / "build.ninja", args.evidence / "build-commands.ninja")
        observation.run(
            "cpp-build", ["cmake", "--build", str(build), "--parallel", "2"], env, 1800
        )
        observation.run("cpp-install", ["cmake", "--install", str(build)], env, 300)
        env.update(ARROW_HOME=str(prefix), CMAKE_PREFIX_PATH=str(prefix))
        for key, directory in (
            ("LD_LIBRARY_PATH", prefix / "lib"),
            ("PKG_CONFIG_PATH", prefix / "lib/pkgconfig"),
        ):
            env[key] = os.pathsep.join(filter(None, (str(directory), env.get(key))))
        for variable in frontend["bundle_defaults"]:
            env.pop(variable, None)
        observation.run(
            "pyarrow-source-build",
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--no-binary",
                "pyarrow",
                "--verbose",
                str(python_archive),
            ],
            env,
            900,
        )
        probe = """import io, tempfile, pathlib, json
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
assert pa.__version__ == '26.0.0', pa.__version__
t = csv.read_csv(io.BytesIO(b'x\\n1\\n2\\n3\\n'))
assert pc.sum(t['x']).as_py() == 6
sink = pa.BufferOutputStream()
with ipc.new_stream(sink, t.schema) as writer: writer.write_table(t)
assert ipc.open_stream(sink.getvalue()).read_all().equals(t)
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory) / 'data.parquet'
    pq.write_table(t, path, compression='zstd')
    assert pq.read_table(path).equals(t)
    assert ds.dataset(directory, format='parquet').to_table().equals(t)
for codec in ('brotli', 'bz2', 'lz4_frame', 'snappy', 'gzip', 'zstd'):
    assert pa.Codec.is_available(codec), codec
assert 'mimalloc' in pa.supported_memory_backends()
print(json.dumps({'version':pa.__version__, 'extension':pa.lib.__file__, 'cpp_version':pa.cpp_version,
                  'memory_backends':pa.supported_memory_backends(), 'csv_ipc_parquet_dataset_compute':True}))
"""
        observation.run("installed-native-probe", [python, "-I", "-c", probe], env, 120)
        installed = json.loads(
            (args.evidence / "installed-native-probe.log").read_text()
        )
        package = Path(installed["extension"]).resolve().parent
        observation.run(
            "native-linkage",
            ["ldd", *map(str, native_objects(package, venv))],
            env,
            120,
        )
        linkage = (args.evidence / "native-linkage.log").read_text()
        validate_linkage(linkage, prefix)
        with observation.lock:
            observation.receipt["outcome"] = "passed_scoped_source_build"
    except BaseException as exc:
        with observation.lock:
            observation.receipt.update(
                outcome="failed_or_resource_limited", error=repr(exc)
            )
        raise
    finally:
        observation.stop.set()
        monitor.join()
        observation.receipt.update(
            elapsed_seconds=round(time.monotonic() - observation.started, 2),
            free_bytes_after=shutil.disk_usage(args.workspace).free,
            monitor_errors=observation.monitor_errors,
        )
        with (args.evidence / "retained-directory-sizes.txt").open("w") as output:
            subprocess.run(  # noqa: S603 - fixed SDK executable and experiment-owned path
                [
                    shutil.which("du") or "/usr/bin/du",
                    "-h",
                    "--max-depth=2",
                    str(args.workspace),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if observation.monitor_errors:
            observation.receipt["outcome"] = "failed_observation"
        with observation.lock:
            observation.save_locked()
        if observation.monitor_errors:
            raise RuntimeError(
                "Filesystem sampling failed: " + repr(observation.monitor_errors)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
