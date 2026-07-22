#!/usr/bin/env python3
"""Validate the independently generated LaTeXML accessibility preview."""

from __future__ import annotations

from pathlib import Path
import re
import sys


def main() -> None:
    """Require a minimal semantic document without LaTeXML error markers."""
    html = Path(sys.argv[1]).read_text(errors="replace")
    errors: list[str] = []
    for label, pattern in {
        "HTML document": r"<html\b",
        "title": r"<title\b[^>]*>.*?</title>",
        "abstract": r"abstract",
        "semantic section heading": r"<h[1-6]\b",
        "bibliography": r"bibliography|References",
    }.items():
        if not re.search(pattern, html, re.IGNORECASE | re.DOTALL):
            errors.append(f"missing {label}")
    if re.search(r"ltx_(?:ERROR|fatal)", html, re.IGNORECASE):
        errors.append("LaTeXML emitted an error marker")
    if errors:
        raise SystemExit("\n".join(errors))
    print("semantic HTML validation: pass")


if __name__ == "__main__":
    main()
