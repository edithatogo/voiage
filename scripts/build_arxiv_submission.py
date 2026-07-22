#!/usr/bin/env python3
"""Compile canonical LaTeX and create a deterministic, non-submitting archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = ROOT / "build" / "arxiv"
PACKAGE = BUILD / "package"
SOURCE_SUFFIXES = {
    ".tex",
    ".bib",
    ".bbl",
    ".cls",
    ".sty",
    ".bst",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".eps",
    ".ps",
}


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    """Run a required build command."""
    # Commands are repository-owned fixed argument lists, never user shell text.
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def copy_sources() -> None:
    """Copy only arXiv-compatible authored source and referenced assets."""
    for source in sorted(PAPER.rglob("*")):
        if source.is_file() and source.suffix.lower() in SOURCE_SUFFIXES:
            destination = PACKAGE / source.relative_to(PAPER)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    generated_bbl = BUILD / "main.bbl"
    if generated_bbl.exists():
        shutil.copy2(generated_bbl, PACKAGE / "main.bbl")


def deterministic_archive(epoch: int) -> tuple[Path, str]:
    """Create a byte-reproducible source archive and checksum."""
    archive = BUILD / "voiage-arxiv-source.tar.gz"
    archive.unlink(missing_ok=True)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=9, mtime=epoch
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for path in sorted(PACKAGE.rglob("*")):
            relative = path.relative_to(PACKAGE)
            info = tar.gettarinfo(path, arcname=str(relative))
            info.mtime = epoch
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if path.is_file():
                with path.open("rb") as source:
                    tar.addfile(info, source)
            else:
                tar.addfile(info)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = BUILD / "voiage-arxiv-source.tar.gz.sha256"
    checksum.write_text(f"{digest}  {archive.name}\n")
    return archive, digest


def main() -> None:
    """Build and report the non-submitting arXiv readiness package."""
    if "--clean" in sys.argv:
        shutil.rmtree(BUILD, ignore_errors=True)
        return
    for tool in ("pdflatex", "latexmk"):
        if shutil.which(tool) is None:
            raise SystemExit(f"missing required tool: {tool}")
    shutil.rmtree(BUILD, ignore_errors=True)
    PACKAGE.mkdir(parents=True)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    os.environ["SOURCE_DATE_EPOCH"] = str(epoch)
    run(["latexmk", f"-outdir={BUILD}", "main.tex"], cwd=PAPER)
    copy_sources()
    run([sys.executable, str(ROOT / "scripts/validate_arxiv.py"), str(PACKAGE)])
    archive, digest = deterministic_archive(epoch)
    manifest = json.loads((PAPER / "readiness-manifest.json").read_text())
    report = {
        "schema_version": "voiage.arxiv-readiness.v2",
        "status": "ready_for_human_review",
        "top_level_tex": "main.tex",
        "preview_pdf": str((BUILD / "main.pdf").relative_to(ROOT)),
        "archive": str(archive.relative_to(ROOT)),
        "sha256": digest,
        "submission_performed": False,
        "joss_submission": manifest["joss_submission"],
        "human_gates": manifest["human_gates"],
    }
    (BUILD / "arxiv-readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
