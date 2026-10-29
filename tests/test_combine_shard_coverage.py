"""Exercise native coverage combination and the existing policy gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from coverage import CoverageData

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/combine_shard_coverage.py"
PREPARE = Path(__file__).resolve().parents[1] / "scripts/prepare_shard_receipts.py"


def run_command(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def scenario(
    tmp_path: Path, *, branch: bool = False, arcs: bool = False
) -> tuple[list[str], list[dict[str, object]], Path]:
    repo = tmp_path / "repo"
    (repo / "voiage").mkdir(parents=True)
    source = repo / "voiage/example.py"
    source.write_text("first = 1\nsecond = 2\n", encoding="utf-8")
    lock = repo / "uv.lock"
    lock.write_text("fixture lock\n", encoding="utf-8")
    config = repo / "pyproject.toml"
    config.write_text(
        f"[tool.coverage.run]\nbranch = {str(branch).lower()}\n", encoding="utf-8"
    )
    run_command("git", "init", "-q", cwd=repo)
    run_command(
        "git", "add", "voiage/example.py", "uv.lock", "pyproject.toml", cwd=repo
    )
    run_command(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
        cwd=repo,
    )
    head = run_command("git", "rev-parse", "HEAD", cwd=repo)
    identity = {
        "source_head": head,
        "lock_digest": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "interpreter": "CPython 3.15.1",
        "features": "ci,dev",
        "coverage_config_digest": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    expected = tmp_path / "identity.json"
    expected.write_text(json.dumps(identity), encoding="utf-8")
    shards: list[dict[str, object]] = []
    for index in range(2):
        report = tmp_path / f"report-{index}.json"
        data_path = tmp_path / f".coverage.shard{index}"
        data = CoverageData(basename=str(data_path))
        if arcs:
            data.add_arcs({str(source): {(-1, 1), (1, 2)} if index == 0 else {(2, -1)}})
        else:
            data.add_lines({str(source): {index + 1}})
        data.write()
        run_command(
            sys.executable,
            "-m",
            "coverage",
            "json",
            "--rcfile",
            str(config),
            "--data-file",
            str(data_path),
            "-o",
            str(report),
            cwd=repo,
        )
        shards.append(
            {
                "index": index,
                "conclusion": "success",
                "exit_status": 0,
                "identity": identity.copy(),
                "coverage": str(report),
                "coverage_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "native_data": str(data_path),
                "native_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"expected_shards": 2, "identity": identity, "shards": shards}),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "aggregate_percent": 100,
                "critical_modules": {},
                "changed_line_percent": 100,
                "changed_branch_percent": 100,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "policy-result.json"
    args = [
        sys.executable,
        str(SCRIPT),
        "--manifest",
        str(manifest),
        "--expected-identity",
        str(expected),
        "--expected-shards",
        "2",
        "--repo",
        str(repo),
        "--base",
        head,
        "--source-head",
        head,
        "--policy",
        str(policy),
        "--output",
        str(output),
    ]
    return args, shards, output


def test_real_native_coverage_combination_reaches_policy(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path)
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_altered_native_coverage_never_produces_green(tmp_path: Path) -> None:
    args, shards, output = scenario(tmp_path)
    Path(str(shards[0]["native_data"])).write_bytes(b"tampered")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not output.exists()


def test_real_coverage_policy_failure_is_propagated(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path)
    policy = Path(args[args.index("--policy") + 1])
    data = json.loads(policy.read_text(encoding="utf-8"))
    data["aggregate_percent"] = 101
    policy.write_text(json.dumps(data), encoding="utf-8")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False


def test_changed_coverage_configuration_is_rejected(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path)
    repo = Path(args[args.index("--repo") + 1])
    (repo / "pyproject.toml").write_text(
        "[tool.coverage.run]\nbranch = true\n", encoding="utf-8"
    )
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not output.exists()


def test_line_only_data_cannot_satisfy_branch_coverage(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path, branch=True)
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not output.exists()


def test_native_branch_data_combines_and_reaches_policy(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path, branch=True, arcs=True)
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_dirty_source_cannot_pass_on_same_head(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path)
    repo = Path(args[args.index("--repo") + 1])
    with (repo / "voiage/example.py").open("a", encoding="utf-8") as source:
        source.write("# uncommitted change\n")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not output.exists()


def test_report_must_match_its_native_data(tmp_path: Path) -> None:
    args, _, output = scenario(tmp_path)
    manifest_path = Path(args[args.index("--manifest") + 1])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    report_path = Path(shard["coverage"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    next(iter(report["files"].values()))["executed_lines"] = [2]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    shard["coverage_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert not output.exists()


def test_prepared_receipts_drive_actual_combination(tmp_path: Path) -> None:
    args, shards, output = scenario(tmp_path)
    repo = Path(args[args.index("--repo") + 1])
    root = tmp_path / "artifacts"
    for index, shard in enumerate(shards):
        directory = root / f"shard-{index}"
        directory.mkdir(parents=True)
        shutil.copyfile(str(shard["coverage"]), directory / "coverage.json")
        shutil.copyfile(str(shard["native_data"]), directory / "coverage.data")
        prepared = subprocess.run(
            [
                sys.executable,
                str(PREPARE),
                "prepare",
                "--repo",
                str(repo),
                "--root",
                str(root),
                "--index",
                str(index),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stderr
    collected = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "collect",
            "--repo",
            str(repo),
            "--root",
            str(root),
            "--matrix-result",
            "success",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert collected.returncode == 0, collected.stderr
    args[args.index("--manifest") + 1] = str(root / "manifest.json")
    args[args.index("--expected-identity") + 1] = str(root / "expected-identity.json")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
