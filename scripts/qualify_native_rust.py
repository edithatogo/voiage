"""Build authenticated Rust and LLVM sources and probe the installed compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from scripts.observe_native_stage import StageObserver
from scripts.qualify_source_vendor import digest, unpack

TARGET = "x86_64-unknown-linux-gnu"
ENTRIES = ("rust-stable", "rust-nightly")


def bootstrap_config(work: Path, entry: str) -> dict[str, Any]:
    """Freeze one optimized stage2 Linux host without narrowing LLVM backends."""
    if entry not in ENTRIES:
        raise ValueError("Unsupported compiler entry")
    sdk = work / "bootstrap"
    return {
        "change-id": "ignore",
        "build": {
            "build": TARGET,
            "host": [TARGET],
            "target": [TARGET],
            "build-dir": str(work / "build"),
            "cargo": str(sdk / "bin/cargo"),
            "rustc": str(sdk / "bin/rustc"),
            "python": sys.executable,
            "vendor": True,
            "locked-deps": True,
            "extended": True,
            "tools": ["cargo", "rustdoc", "src"],
            "docs": False,
            "submodules": False,
        },
        "llvm": {"download-ci-llvm": False, "link-jobs": 1, "ninja": True},
        "rust": {
            "download-rustc": False,
            "channel": "stable" if entry == "rust-stable" else "nightly",
            "optimize": True,
            "debug": False,
            "debuginfo-level": 0,
            "codegen-units": 4,
            "codegen-units-std": 1,
            "incremental": False,
        },
        "install": {"prefix": str(work / "install"), "sysconfdir": "etc"},
    }


def validate_config(config: dict[str, Any], work: Path) -> None:
    """Reject binary substitutions and prefixes that cannot prove source output."""
    if (
        config["llvm"]["download-ci-llvm"] is not False
        or config["rust"]["download-rustc"] is not False
    ):
        raise ValueError("Rust and LLVM must be source-built")
    if (
        config["install"]["prefix"] != str(work / "install")
        or not config["build"]["vendor"]
        or not config["build"]["locked-deps"]
    ):
        raise ValueError("Unqualified source build configuration")


def validate_installed_version(text: str, entry: str) -> None:
    """Reject the stage0 SDK and require an installed LLVM-enabled compiler."""
    version = "1.99.1" if entry == "rust-stable" else "1.100.0-nightly"
    if not text.startswith("rustc " + version + " ") or "LLVM version:" not in text:
        raise ValueError("Unexpected installed compiler identity")


def write_config(config: dict[str, Any]) -> str:
    """Serialize the constrained scalar/array TOML profile without a new dependency."""
    lines = []
    for section, value in config.items():
        if not isinstance(value, dict):
            lines.append(f"{section} = {json.dumps(value)}")
        else:
            lines.append(f"\n[{section}]")
            lines.extend(f"{key} = {json.dumps(item)}" for key, item in value.items())
    return "\n".join(lines) + "\n"


class Build:
    """Retain each real command and its bounded source-stage receipt."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work = args.work.resolve()
        self.evidence = args.evidence.resolve()
        self.work.mkdir()
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.env = dict(
            os.environ,
            CARGO_HOME=str(self.work / "cargo-home"),
            CARGO_NET_OFFLINE="true",
            CARGO_NET_RETRY="0",
            HOME=os.environ["HOME"],
            CARGO_BUILD_JOBS="4",
            CMAKE_BUILD_PARALLEL_LEVEL="4",
        )
        for key in (
            "RUSTC",
            "RUSTDOC",
            "RUSTUP_TOOLCHAIN",
            "CARGO_TARGET_DIR",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
        ):
            self.env.pop(key, None)
        self.receipt: dict[str, Any] = {
            "entry": args.entry,
            "outcome": "running",
            "steps": [],
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),  # noqa: S607 - qualified Git CLI
            "run_url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'local')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', 'local')}",
            "python": sys.version,
            "jobs": 4,
            "stage": 2,
            "cache_state": "cold",
            "full_hpc_matrix_qualified": False,
            "compiler_source_build_accepted": False,
            "llvm_source_build_accepted": False,
        }

    def save(self) -> None:
        """Persist a checkpoint before and after each stage."""
        self.receipt["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        (self.evidence / "receipt.json").write_text(
            json.dumps(self.receipt, indent=2) + "\n"
        )

    def run(
        self,
        name: str,
        command: list[str],
        *,
        cwd: Path | None = None,
        offline: bool = True,
    ) -> None:
        """Use kernel network isolation only after authenticated prefetch finishes."""
        timeout = min(14400, 19800 - (time.monotonic() - self.started))
        if timeout <= 0:
            raise RuntimeError("Compiler feasibility envelope exhausted")
        row = {"name": name, "argv": command, "network_isolated": offline}
        self.receipt["steps"].append(row)
        self.save()
        print(f"Starting {name}", flush=True)
        if offline:
            preserved = "PATH,HOME,CARGO_HOME,CARGO_NET_OFFLINE,CARGO_NET_RETRY,CARGO_BUILD_JOBS,CMAKE_BUILD_PARALLEL_LEVEL,RUSTC"
            monitor = [
                "sudo",
                f"--preserve-env={preserved}",
                sys.executable,
                "-m",
                "scripts.observe_native_stage",
                "--evidence",
                str(self.evidence / name),
                "--workspace",
                str(self.work),
                "--cwd",
                str(cwd or Path.cwd()),
                "--timeout",
                str(timeout),
                "--uid",
                str(os.getuid()),
                "--gid",
                str(os.getgid()),
                "--",
                *command,
            ]
            result = subprocess.run(monitor, env=self.env, check=False)  # noqa: S603 - fixed observer CLI and authenticated source argv
            code = result.returncode
        else:
            observer = StageObserver(self.evidence / name, self.work, timeout=timeout)
            online_env = dict(self.env)
            online_env.pop("CARGO_NET_OFFLINE", None)
            code = observer.run(command, env=online_env, cwd=cwd)
        row["returncode"] = code
        self.save()
        print(f"Completed {name}: {code}", flush=True)
        if code:
            raise RuntimeError(f"{name} failed with exit {code}")

    def execute(self) -> None:
        """Authenticate, build stage2 from source and run an installed Cargo probe."""
        inputs = json.loads(
            Path("packaging/current/source-vendor-inputs.json").read_text()
        )
        entry = inputs["entries"][self.args.entry]
        self.receipt["source_inputs"] = entry
        self.run(
            "source-vendor",
            [
                sys.executable,
                "-m",
                "scripts.qualify_source_vendor",
                "--entry",
                self.args.entry,
                "--work",
                str(self.work / "source-work"),
                "--output",
                str(self.evidence / "source-qualification"),
                "--cargo",
                str(self.args.cargo),
                "--rustc",
                str(self.args.rustc),
            ],
            offline=False,
        )
        source = next((self.work / "source-work/unpacked").iterdir())
        archive = (
            self.work
            / "source-work/archives"
            / entry["bootstrap"]["url"].rsplit("/", 1)[-1]
        )
        sdk = unpack(archive, self.work / "bootstrap-source")
        self.run(
            "bootstrap-install",
            [
                "bash",
                str(sdk / "install.sh"),
                "--prefix=" + str(self.work / "bootstrap"),
                "--components=rustc,cargo,rust-std-" + TARGET,
                "--disable-ldconfig",
            ],
        )
        self.run(
            "bootstrap-identity",
            [str(self.work / "bootstrap/bin/rustc"), "--version", "--verbose"],
        )
        bootstrap_version = (
            self.evidence / "bootstrap-identity/command.log"
        ).read_text()
        expected = "1.98.0" if self.args.entry == "rust-stable" else "1.99.0-beta.1"
        if not bootstrap_version.startswith("rustc " + expected + " "):
            raise ValueError("Unexpected authenticated bootstrap identity")
        config = bootstrap_config(self.work, self.args.entry)
        validate_config(config, self.work)
        text = write_config(config)
        (source / "bootstrap.toml").write_text(text)
        (self.evidence / "bootstrap.toml").write_text(text)
        config_dir = source / ".cargo"
        config_dir.mkdir(exist_ok=True)
        original = config_dir / "config.toml"
        if original.exists():
            shutil.copyfile(original, self.evidence / "upstream-cargo-config.toml")
        shutil.copyfile(
            self.evidence / "source-qualification/vendor-config.toml", original
        )
        self.receipt["derived_cargo_config_sha256"] = digest(original)
        self.receipt["bootstrap_config_sha256"] = hashlib.sha256(
            text.encode()
        ).hexdigest()
        self.run(
            "network-control",
            [
                sys.executable,
                "-c",
                "import os,socket;assert os.geteuid()!=0;s=socket.socket();s.settimeout(1);assert s.connect_ex(('1.1.1.1',443))!=0;print('unprivileged isolated network confirmed')",
            ],
        )
        self.run(
            "host-tools",
            [
                "bash",
                "-c",
                "set -e; gcc --version; g++ --version; cmake --version; ninja --version; unshare --version; /usr/bin/time --version",
            ],
        )
        x = [sys.executable, str(source / "x.py")]
        self.run(
            "llvm-source",
            [*x, "build", "--stage", "0", "--jobs", "4", "src/llvm-project"],
            cwd=source,
        )
        caches = list((self.work / "build").rglob("CMakeCache.txt"))
        llvm_caches = [p for p in caches if "llvm" in p.parts]
        if not llvm_caches:
            raise ValueError("No source-built LLVM CMake evidence")
        for index, p in enumerate(llvm_caches):
            shutil.copyfile(p, self.evidence / f"llvm-cache-{index}.txt")
        self.run(
            "rust-stage2-source",
            [
                *x,
                "build",
                "--stage",
                "2",
                "--jobs",
                "4",
                "compiler/rustc",
                "library/std",
                "src/tools/cargo",
                "src/tools/rustdoc",
            ],
            cwd=source,
        )
        self.run(
            "install-stage2", [*x, "install", "--stage", "2", "--jobs", "4"], cwd=source
        )
        prefix = self.work / "install"
        self.run(
            "installed-identity", [str(prefix / "bin/rustc"), "--version", "--verbose"]
        )
        validate_installed_version(
            (self.evidence / "installed-identity/command.log").read_text(),
            self.args.entry,
        )
        probe = self.work / "installed-probe"
        (probe / "src").mkdir(parents=True)
        (probe / "Cargo.toml").write_text(
            '[package]\nname="native-compiler-probe"\nversion="0.1.0"\nedition="2024"\n[workspace]\n'
        )
        (probe / "src/main.rs").write_text(
            'fn main(){let n=[[10.0_f64,20.0],[30.0,10.0]];let perfect=n.iter().map(|r|r[0].max(r[1])).sum::<f64>()/2.0;let current=((n[0][0]+n[1][0])/2.0).max((n[0][1]+n[1][1])/2.0);assert!((perfect-current-5.0).abs()<1e-12);println!("EVPI={}",perfect-current);}\n'
        )
        self.env["RUSTC"] = str(prefix / "bin/rustc")
        self.run(
            "installed-cargo-compile-link",
            [
                str(prefix / "bin/cargo"),
                "run",
                "--offline",
                "--release",
                "--manifest-path",
                str(probe / "Cargo.toml"),
            ],
            cwd=probe,
        )
        self.run("installed-linkage", ["ldd", str(prefix / "bin/rustc")])
        if "not found" in (self.evidence / "installed-linkage/command.log").read_text():
            raise ValueError("Installed compiler has unresolved native libraries")
        qualification = json.loads(
            (self.evidence / "source-qualification/qualification.json").read_text()
        )
        if any(
            digest(source / path) != sha for path, sha in qualification["locks"].items()
        ):
            raise ValueError("Native build changed a qualified lock")
        self.receipt.update(
            outcome="passed_native_compiler",
            compiler_source_build_accepted=True,
            llvm_source_build_accepted=True,
            installed_prefix=str(prefix),
        )


def main() -> int:
    """Keep failures red and retain source-stage receipts through interruption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=ENTRIES, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 15, 1) or sys.platform != "linux":
        raise RuntimeError("Select repo-local Python 3.15.1 on Linux")
    build = Build(args)
    try:
        build.execute()
    except BaseException as error:
        build.receipt.update(outcome="failed_or_resource_limited", error=repr(error))
        raise
    finally:
        build.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
