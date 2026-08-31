#!/usr/bin/env python3
"""Run bounded, serial local feedback without weakening the ordinary test gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from defusedxml import ElementTree


def fingerprint(
    root: Path, files: list[str], targets: list[str], packages: list[str]
) -> str:
    """Invalidate on environment, inventory, config and non-Python resource changes."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                str(root.resolve()),
                sys.version,
                platform.platform(),
                targets,
                sorted(packages),
            ]
        ).encode()
    )
    for name in sorted(set(files)):
        digest.update(name.encode() + b"\0")
        path = root / name
        # Testmon observes Python source changes; resources and native code need
        # an explicit reset because Python line coverage cannot track them.
        if (
            path.suffix != ".py"
            or path.name == "conftest.py"
            or name == "scripts/test_acceleration.py"
        ):
            digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def invalidate_cache(cache: Path, current: str) -> bool:
    """Discard only this runner's database if the completed-run signature differs."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "fingerprint"
    if manifest.is_file() and manifest.read_text() == current:
        return False
    for suffix in ("", "-wal", "-shm", "-journal"):
        (cache / ("testmon" + suffix)).unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)
    return True


def validate_targets(root: Path, targets: list[str]) -> list[str]:
    """Allow test paths, never arbitrary pytest options or node selections."""
    for target in targets:
        path = Path(target)
        if path.is_absolute() or target.startswith("-") or "::" in target:
            raise ValueError(
                "targets must be repository test paths, not pytest options"
            )
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root / "tests") or not resolved.exists():
            raise ValueError("targets must exist below repository tests/")
    return targets


def main() -> int:
    """Execute one isolated profile and retain measured, non-authorizing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", choices=("ordinary", "testmon", "gremlins", "ctrace", "sysmon")
    )
    parser.add_argument("targets", nargs="*", default=["tests"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        targets = validate_targets(root, args.targets)
    except ValueError as error:
        return parser.error(str(error))
    cache = root / ".conductor/local/test-acceleration"
    cache.mkdir(parents=True, exist_ok=True)
    # A separate process lock prevents serial invocations from corrupting the
    # shared selective database. A stale lock fails closed; inspect before removal.
    lock = cache / "running"
    try:
        lock.mkdir()
    except FileExistsError:
        parser.error(
            f"profile already running or stale lock requires inspection: {lock}"
        )
    try:
        return run_profile(root, cache, args.profile, targets)
    finally:
        lock.rmdir()


def run_profile(root: Path, cache: Path, profile: str, targets: list[str]) -> int:
    """Run pytest with fixed plugin and trace settings; record actual JUnit counts."""
    # Missing terminal evidence is safer than a stale success after a crash.
    (cache / f"{profile}.json").unlink(missing_ok=True)
    if profile == "gremlins":
        (root / "coverage/gremlins/gremlins.json").unlink(missing_ok=True)
    total_started = time.perf_counter()
    packages = sorted(
        f"{p.metadata['Name']}=={p.version}" for p in importlib.metadata.distributions()
    )
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        cwd=root,
        text=True,
    ).splitlines()
    current = fingerprint(root, files, targets, packages)
    reset = invalidate_cache(cache, current) if profile == "testmon" else False
    env = os.environ.copy()
    for key in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "COVERAGE_PROCESS_START",
        "COVERAGE_CORE",
        "VOIAGE_TEST_SHARD_INDEX",
        "VOIAGE_TEST_SHARD_COUNT",
    ):
        env.pop(key, None)
    if profile in ("testmon", "gremlins"):
        env["COVERAGE_CORE"] = "ctrace"
    if profile == "testmon":
        # Testmon switches dynamic contexts; sysmon cannot support that safely.
        env["COVERAGE_CORE"] = "ctrace"
        (cache / "fingerprint").unlink(missing_ok=True)
    env["TESTMON_DATAFILE"] = str(cache / "testmon")
    env["PYTHONPATH"] = str(root)
    junit = cache / f"{profile}.xml"
    junit.unlink(missing_ok=True)
    pytest_args = [
        "-o",
        "addopts=",
        "-p",
        "no:pytest_cov",
        *targets,
        "--junitxml",
        str(junit),
    ]
    if profile != "gremlins":
        pytest_args += ["-p", "no:gremlins"]
    if profile != "testmon":
        pytest_args += ["-p", "no:pytest-testmon"]
    prefix = [sys.executable, "-m", "pytest"]
    if profile == "testmon":
        pytest_args += ["--testmon"]
    if profile == "gremlins":
        # Never let gremlins discover the entire runtime or mutate arbitrary paths.
        pytest_args = [
            "-o",
            "addopts=",
            "-p",
            "no:pytest_cov",
            "tests/test_mutation_score.py",
            "--junitxml",
            str(junit),
            "--gremlins",
            "--gremlin-targets=voiage/mutation_policy.py",
            "--gremlin-operators=comparison,boolean",
            "--gremlin-workers=1",
            "-p",
            "no:pytest-testmon",
            "--gremlin-report=json",
        ]
    if profile in ("ctrace", "sysmon"):
        env["COVERAGE_CORE"] = profile
        (cache / "coverage.ini").write_text("[run]\n")
        prefix = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(cache / "coverage.ini"),
            "--debug=core",
            "--branch",
            "--include=*/voiage/mutation_policy.py",
            "--data-file",
            str(cache / f".coverage-{profile}"),
            "-m",
            "pytest",
        ]
    command = [*prefix, *pytest_args]
    started = time.perf_counter()
    result = subprocess.run(  # noqa: S603 - fixed interpreter/profile and validated test paths
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = result.stdout + result.stderr
    (cache / f"{profile}.log").write_text(output)
    print(output)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if junit.exists():
        for suite in ElementTree.parse(junit).getroot().iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
    # Exit 5 only means cached no-selection when testmon reused a successful
    # baseline; never translate failed collection or an empty cold run to success.
    cached_empty = (
        profile == "testmon"
        and not reset
        and result.returncode in (0, 5)
        and counts["tests"] == 0
        and counts["errors"] == 0
    )
    passed = junit.exists() and (
        (result.returncode == 0 and counts["tests"] > 0) or cached_empty
    )
    collected = re.search(r"collected (\d+) items?", output)
    if profile == "testmon":
        if passed:
            (cache / "fingerprint").write_text(current)
        else:
            (cache / "fingerprint").unlink(missing_ok=True)
    evidence = {
        "profile": profile,
        "command": command,
        "wall_seconds": round(elapsed, 4),
        "total_wall_seconds": round(time.perf_counter() - total_started, 4),
        "collected": int(collected.group(1)) if collected else None,
        "exit_code": result.returncode,
        "cached_empty": cached_empty,
        "cache_invalidated": reset,
        "counts": counts,
        "python": sys.version,
        "packages": packages,
        "fingerprint": current,
        "local_feedback_only": True,
        "coverage_core": env.get("COVERAGE_CORE"),
    }
    if profile == "gremlins":
        report_path = root / "coverage/gremlins/gremlins.json"
        if report_path.is_file():
            evidence["mutation_summary"] = json.loads(report_path.read_text())[
                "summary"
            ]
    (cache / f"{profile}.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "packages"}))
    return 0 if passed else (result.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
