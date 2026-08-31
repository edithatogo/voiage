#!/usr/bin/env python3
"""Validate the independently generated LaTeXML accessibility preview."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


class _Graphics(HTMLParser):
    """Collect the canonical manuscript's graphics, not page-logo images."""

    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "img" and "ltx_graphics" in (values.get("class") or "").split():
            self.sources.append(values.get("src") or "")


def validate_html(path: Path) -> list[str]:
    """Require semantic content and portable, present manuscript graphics."""
    html = path.read_text(errors="replace")
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
    if "ltx_missing_image" in html:
        errors.append("LaTeXML omitted a manuscript image")
    graphics = _Graphics()
    graphics.feed(html)
    if not graphics.sources:
        errors.append("missing canonical manuscript figure")
    for source in graphics.sources:
        url = urlsplit(source)
        relative = Path(unquote(url.path))
        target = (path.parent / relative).resolve()
        if (
            not source.strip()
            or url.scheme
            or url.netloc
            or relative.is_absolute()
            or not target.is_relative_to(path.parent.resolve())
            or not target.is_file()
            or target.stat().st_size == 0
        ):
            errors.append(
                "manuscript graphic must resolve to a nonempty local artifact"
            )
    return errors


def main() -> None:
    """Reject incomplete HTML even when the converter exits successfully."""
    errors = validate_html(Path(sys.argv[1]))
    if errors:
        raise SystemExit("\n".join(errors))
    print("semantic HTML validation: pass")


if __name__ == "__main__":
    main()
