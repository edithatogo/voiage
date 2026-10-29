"""Exercise the real coverage gate rather than only its policy declarations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_coverage_policy.py"


def test_coverage_gate_rejects_missing_input_and_wrong_head(tmp_path: Path) -> None:
    """A missing report or mismatched checkout cannot produce a successful gate."""
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = tmp_path / "coverage.json"
    policy = tmp_path / "policy.json"
    output = tmp_path / "result.json"
    policy.write_text(
        json.dumps(
            {
                "aggregate_percent": 90,
                "critical_modules": {},
                "changed_line_percent": 100,
                "changed_branch_percent": 100,
            }
        ),
        encoding="utf-8",
    )

    def run(source_head: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo",
                str(ROOT),
                "--coverage",
                str(report),
                "--policy",
                str(policy),
                "--base",
                head,
                "--source-head",
                source_head,
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    assert run(head).returncode != 0
    assert not output.exists()
    report.write_text(
        json.dumps({"totals": {"percent_covered": 100}, "files": {}}),
        encoding="utf-8",
    )
    assert run(head).returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert run("0" * 40).returncode == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["exact_source_head"] is False
