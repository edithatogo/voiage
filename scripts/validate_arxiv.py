#!/usr/bin/env python3
"""Fail-closed checks derived from arXiv's source-submission guidance."""

from __future__ import annotations

from pathlib import Path
import re
import sys

ALLOWED_NAME = re.compile(r"^[A-Za-z0-9_+.,=-]+$")
FORBIDDEN_SUFFIXES = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz"}
ABSOLUTE_PATH = re.compile(r"(?:/Users/|/Volumes/|/home/|[A-Za-z]:\\)")


def main() -> None:
    """Validate a canonical or transformed arXiv source tree."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "paper").resolve()
    errors: list[str] = []
    main_tex = root / "main.tex"
    if not main_tex.exists() or "\\documentclass" not in main_tex.read_text(
        errors="replace"
    ):
        errors.append("main.tex must exist and contain \\documentclass")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(not ALLOWED_NAME.fullmatch(part) for part in relative.parts):
            errors.append(f"arXiv-incompatible filename: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated file must not be submitted: {relative}")
        if path.suffix.lower() in {".tex", ".bib", ".sty", ".cls"}:
            text = path.read_text(errors="replace")
            if ABSOLUTE_PATH.search(text):
                errors.append(f"local absolute path in source: {relative}")
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if total > 50 * 1024 * 1024:
        errors.append(f"source package exceeds 50 MiB: {total} bytes")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"arXiv source validation: pass ({total} bytes)")


if __name__ == "__main__":
    main()
