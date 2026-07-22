"""Build a reviewable, non-submitting arXiv source package for voiage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "arxiv"
PACKAGE = OUT / "package"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, check=True, capture_output=True)


def version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    result = subprocess.run([executable, "--version"], text=True, capture_output=True)
    return (
        (result.stdout or result.stderr).splitlines()[0]
        if result.returncode == 0
        else None
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(PACKAGE, ignore_errors=True)
    PACKAGE.mkdir(parents=True)

    required = {tool: version(tool) for tool in ("quarto", "pdflatex")}
    missing = [tool for tool, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing required arXiv build tools: {', '.join(missing)}")

    rendered = OUT / "rendered"
    shutil.rmtree(rendered, ignore_errors=True)
    rendered.mkdir()
    latex = OUT / "latex"
    shutil.rmtree(latex, ignore_errors=True)
    latex.mkdir()
    run(
        [
            "quarto",
            "render",
            "paper.md",
            "--to",
            "latex",
            "--metadata",
            "bibliography=paper.bib",
            "--output-dir",
            str(latex),
        ]
    )
    run(
        [
            "quarto",
            "render",
            "paper.md",
            "--to",
            "pdf",
            "--pdf-engine",
            "pdflatex",
            "--metadata",
            "bibliography=paper.bib",
            "--output-dir",
            str(rendered),
        ]
    )

    pdfs = sorted(rendered.rglob("*.pdf"))
    texs = sorted(latex.rglob("*.tex"))
    if len(pdfs) != 1 or len(texs) != 1:
        raise SystemExit(
            f"Expected one PDF and one TeX output, found {len(pdfs)} and {len(texs)}"
        )

    shutil.copy2(texs[0], PACKAGE / "main.tex")
    shutil.copy2(ROOT / "paper.bib", PACKAGE / "references.bib")
    shutil.copy2(pdfs[0], PACKAGE / "voiage-arxiv.pdf")
    shutil.copy2(ROOT / "paper/arxiv/manifest.json", PACKAGE / "manifest.json")

    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    forbidden = re.compile(
        r"(/Users/|/Volumes/|-----BEGIN (?:RSA|OPENSSH|PGP) PRIVATE KEY-----)"
    )
    findings = [
        str(path.relative_to(PACKAGE))
        for path in files
        if forbidden.search(path.read_text(errors="ignore"))
    ]
    if findings:
        raise SystemExit(f"Unsafe local-path or secret content found: {findings}")

    archive = OUT / "voiage-arxiv.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(PACKAGE))

    report = {
        "schema_version": "voiage.arxiv-readiness.v1",
        "status": "ready_for_human_review",
        "required_tools": required,
        "package": str(archive.relative_to(ROOT)),
        "sha256": sha256(archive),
        "files": [str(path.relative_to(PACKAGE)) for path in files],
        "human_gates": json.loads((ROOT / "paper/arxiv/manifest.json").read_text())[
            "human_gates"
        ],
        "submission_performed": False,
        "joss_submission": "out_of_scope_for_this_execution",
    }
    (OUT / "arxiv-readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
