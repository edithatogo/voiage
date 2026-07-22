#!/usr/bin/env python3
"""Audit the review PDF and LaTeX log without treating the PDF as source."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys


def capture(command: list[str]) -> str:
    """Capture output from an assurance command."""
    # Commands are repository-owned fixed argument lists, never user shell text.
    return subprocess.run(  # noqa: S603
        command, check=True, capture_output=True, text=True
    ).stdout


def main() -> None:
    """Reject invalid PDFs, missing fonts, and unresolved LaTeX diagnostics."""
    pdf = Path(sys.argv[1])
    log = Path(sys.argv[2])
    tools = {name: shutil.which(name) for name in ("qpdf", "pdfinfo", "pdffonts")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise SystemExit(f"missing required PDF audit tools: {', '.join(missing)}")
    subprocess.run([str(tools["qpdf"]), "--check", str(pdf)], check=True)  # noqa: S603
    info = capture([str(tools["pdfinfo"]), str(pdf)])
    fonts = capture([str(tools["pdffonts"]), str(pdf)])
    errors: list[str] = []
    pages = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if not pages or int(pages.group(1)) < 1:
        errors.append("PDF has no pages")
    if not re.search(r"^Encrypted:\s+no$", info, re.MULTILINE):
        errors.append("PDF must not be encrypted")
    font_rows = [line for line in fonts.splitlines()[2:] if line.strip()]
    if not font_rows:
        errors.append("PDF contains no detectable fonts")
    for row in font_rows:
        flags = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row)
        if not flags or flags.group(1) != "yes":
            errors.append(f"font is not embedded: {row.strip()}")
    log_text = log.read_text(errors="replace")
    forbidden = {
        "LaTeX compilation error": r"^! LaTeX Error:",
        "undefined citation": r"Citation .+ undefined",
        "undefined reference": r"Reference .+ undefined|There were undefined references",
        "overfull box": r"Overfull \\hbox",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, log_text, re.MULTILINE):
            errors.append(label)
    if errors:
        raise SystemExit("\n".join(errors))
    page_count = pages.group(1) if pages else "0"
    print(f"PDF audit: pass ({page_count} page(s), {len(font_rows)} embedded font(s))")


if __name__ == "__main__":
    main()
