#!/usr/bin/env python3
"""Produce review-only Textstat evidence from the official JOSS PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_arxiv_readability import build_report, extract_text

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Extract the official review PDF and write review-only Textstat evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=ROOT / "build/joss/readability.json",
    )
    args = parser.parse_args()
    pdf = args.pdf if args.pdf.is_absolute() else ROOT / args.pdf
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not pdf.is_file():
        raise SystemExit(f"official JOSS review PDF does not exist: {pdf}")
    report = build_report(extract_text(pdf), pdf.name)
    report["schema_version"] = "voiage.joss-readability.v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "JOSS readability evidence: pass "
        f"({report['counts']['words']} words, "
        f"{report['counts']['sentences']} sentences)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
