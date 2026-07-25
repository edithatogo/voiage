#!/usr/bin/env python3
"""Run the pinned SourceRight citation workflow without editing source files."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCERIGHT = ROOT / ".repo-tools/sourceright"
PANDOC_CITATION = re.compile(r"\[(?P<body>[^\[\]]*@[A-Za-z0-9_:-]+[^\[\]]*)\]")
PANDOC_CITATION_KEY = re.compile(r"@([A-Za-z0-9_:-]+)")


def _run(
    command: list[str],
    *,
    output: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603 - repository-owned fixed commands
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"SourceRight audit command failed ({result.returncode}): "
            + " ".join(command)
            + f"\n{result.stderr}"
        )
    return result


def _source_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sourceright_manuscript(source: str) -> str:
    """Convert Pandoc citations to SourceRight's identifier citation syntax."""

    def replace(match: re.Match[str]) -> str:
        keys = PANDOC_CITATION_KEY.findall(match.group("body"))
        return r"\cite{" + ",".join(keys) + "}"

    return PANDOC_CITATION.sub(replace, source)


def _json_object(path: Path) -> dict[str, Any]:
    loaded: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"SourceRight output is not a JSON object: {path}")
    return loaded


def audit(output_directory: Path) -> dict[str, Any]:
    """Run the pinned read-only SourceRight workflow and retain its evidence."""
    for tool in ("cargo", "pandoc"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"missing JOSS source-audit tool: {tool}")
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)

    csl = output_directory / "references.csl.json"
    manuscript = output_directory / "manuscript.sourceright.tex"
    _run(
        [
            "pandoc",
            "paper.bib",
            "--from=biblatex",
            "--to=csljson",
            f"--output={csl}",
        ]
    )
    manuscript.write_text(
        _sourceright_manuscript((ROOT / "paper.md").read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    cargo = [
        "cargo",
        "run",
        "--quiet",
        "--locked",
        "--manifest-path",
        str(SOURCERIGHT / "Cargo.toml"),
        "--bin",
        "sourceright",
        "--",
    ]
    workspace_parent = output_directory / "workspace"
    workspace_parent.mkdir()
    workspace_result = _run([*cargo, "init", str(workspace_parent)])
    workspace = Path(workspace_result.stdout.strip())
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    shutil.copy2(csl, workspace / "references.csl.json")
    shutil.copy2(
        ROOT / "paper/joss-references.verification.json",
        workspace / "references.verification.json",
    )

    _run(
        [*cargo, "validate-csl", "--json", str(csl)],
        output=output_directory / "validate-csl.json",
    )
    _run(
        [*cargo, "citations", str(manuscript), str(workspace)],
        output=output_directory / "citations.md",
    )
    citation_report = (output_directory / "citations.md").read_text(encoding="utf-8")
    if not re.search(r"^- Issues: 0$", citation_report, re.MULTILINE):
        raise RuntimeError(
            "SourceRight citation reconciliation retained issues; "
            "inspect build/joss/sourceright/citations.md"
        )
    _run(
        [*cargo, "provenance", str(manuscript)],
        output=output_directory / "provenance.json",
    )
    _run(
        [*cargo, "report", "--json", str(workspace)],
        output=output_directory / "reference-report.json",
    )
    _run(
        [*cargo, "citation-sync", "--preview", str(workspace)],
        output=output_directory / "citation-sync-preview.json",
    )
    _run(
        [
            *cargo,
            "journal-screen",
            "--platform",
            "generic-webhook",
            "--manuscript",
            "paper.md",
            str(workspace),
        ],
        output=output_directory / "journal-screen.json",
    )
    journal_screen = _json_object(output_directory / "journal-screen.json")
    reference_report = journal_screen.get("reference_report", {})
    summary = (
        reference_report.get("summary", {})
        if isinstance(reference_report, dict)
        else {}
    )
    warning_count = summary.get("warning_count", 0) if isinstance(summary, dict) else 0
    error_count = summary.get("error_count", 0) if isinstance(summary, dict) else 0
    if not isinstance(warning_count, int) or not isinstance(error_count, int):
        raise TypeError("SourceRight journal-screen summary counts are invalid")
    if error_count:
        raise RuntimeError(
            f"SourceRight journal screening retained {error_count} errors"
        )
    commit = _run(["git", "-C", str(SOURCERIGHT), "rev-parse", "HEAD"]).stdout.strip()
    report: dict[str, Any] = {
        "schema_version": "voiage.joss-sourceright-audit.v1",
        "status": "pass_with_warnings" if warning_count else "pass",
        "structural_status": "pass",
        "mode": "read_only",
        "journal_screen": {
            "status": journal_screen.get("status"),
            "warning_count": warning_count,
            "error_count": error_count,
            "human_source_verification_required": warning_count > 0,
        },
        "sourceright_commit": commit,
        "source_sha256": {
            "paper.md": _source_hash(ROOT / "paper.md"),
            "paper.bib": _source_hash(ROOT / "paper.bib"),
        },
        "outputs": sorted(
            str(path.relative_to(ROOT))
            for path in output_directory.iterdir()
            if path.is_file()
        ),
        "interpretation": (
            "SourceRight provides structural citation and provenance triage; "
            "journal-screen warnings remain explicit until human source "
            "verification is recorded. The audit does not certify claim truth "
            "or source appropriateness."
        ),
    }
    (output_directory / "audit-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    """Run the JOSS SourceRight audit from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "build/joss/sourceright",
    )
    args = parser.parse_args()
    output = (
        args.output_directory
        if args.output_directory.is_absolute()
        else ROOT / args.output_directory
    )
    report = audit(output)
    print(
        "JOSS SourceRight audit: "
        f"{report['status']} ({len(report['outputs'])} evidence files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
