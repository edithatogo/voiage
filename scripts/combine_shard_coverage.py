"""Combine native shard coverage and run the repository coverage policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from coverage import Coverage
from coverage.exceptions import CoverageException
from validate_shard_inputs import load_manifest, validate


def execute(args: argparse.Namespace) -> int:
    """Validate and combine inputs, then invoke the coverage policy."""
    manifest = load_manifest(Path(args.manifest))
    identity = load_manifest(Path(args.expected_identity))
    validate(manifest, identity, int(args.expected_shards))
    if identity.get("source_head") != args.source_head:
        raise ValueError("expected identity does not match source head")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("policy output already exists")
    paths: list[str] = []
    reports: list[Path] = []
    for shard in manifest["shards"]:
        raw = shard.get("native_data")
        digest = shard.get("native_sha256")
        if not isinstance(raw, str) or not isinstance(digest, str):
            raise TypeError("missing native coverage receipt")
        path = Path(raw).resolve(strict=True)
        if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("invalid native coverage artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("native coverage digest mismatch")
        if str(path) in paths:
            raise ValueError("duplicate native coverage artifact")
        paths.append(str(path))
        reports.append(Path(shard["coverage"]).resolve(strict=True))
    repo = Path(args.repo).resolve(strict=True)
    policy = Path(args.policy).resolve(strict=True)
    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=repo,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],  # noqa: S607
        cwd=repo,
        text=True,
    )
    if actual_head != args.source_head or status:
        raise ValueError("checkout must be clean at the expected source head")
    config = repo / "pyproject.toml"
    for key, path in (
        ("lock_digest", repo / "uv.lock"),
        ("coverage_config_digest", config),
    ):
        if identity.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{key} does not match checkout")
    interpreter = f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if identity.get("interpreter") != interpreter:
        raise ValueError("interpreter does not match execution")
    with tempfile.TemporaryDirectory(prefix="voiage-coverage-combine-") as temporary:
        data_file = Path(temporary) / ".coverage"
        report = Path(temporary) / "coverage.json"
        os.chdir(repo)
        for index, native_path in enumerate(paths):
            single = Coverage(data_file=native_path, config_file=str(config))
            single.load()
            if single.get_option("run:branch") and not single.get_data().has_arcs():
                raise ValueError("branch coverage requires native arc data")
            single_report = Path(temporary) / f"shard-{index}.json"
            single.json_report(outfile=str(single_report))
            recorded = json.loads(reports[index].read_text(encoding="utf-8"))
            regenerated = json.loads(single_report.read_text(encoding="utf-8"))
            if recorded.get("files") != regenerated.get("files"):
                raise ValueError("shard report does not match native coverage data")
        coverage = Coverage(data_file=str(data_file), config_file=str(config))
        coverage.combine(data_paths=paths, strict=True, keep=True)
        if coverage.get_option("run:branch") and not coverage.get_data().has_arcs():
            raise ValueError("branch coverage requires native arc data")
        coverage.save()
        coverage.json_report(outfile=str(report))
        result = subprocess.run(  # noqa: S603 - explicit operator arguments
            [
                sys.executable,
                str(Path(__file__).with_name("check_coverage_policy.py")),
                "--coverage",
                str(report),
                "--policy",
                str(policy),
                "--repo",
                str(repo),
                "--base",
                args.base,
                "--source-head",
                args.source_head,
                "--output",
                str(output),
            ],
            check=False,
        )
        return result.returncode


def main() -> int:
    """Run the combined coverage gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "manifest",
        "expected-identity",
        "expected-shards",
        "repo",
        "base",
        "source-head",
        "policy",
        "output",
    ):
        parser.add_argument(f"--{option}", required=True)
    args = parser.parse_args()
    try:
        return execute(args)
    except (
        OSError,
        ValueError,
        TypeError,
        CoverageException,
        subprocess.CalledProcessError,
    ) as error:
        print(f"coverage combination failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
