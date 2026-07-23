#!/usr/bin/env python3
"""Produce deterministic, review-only Textstat evidence from the arXiv PDF."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize PDF extraction while joining words hyphenated across lines."""
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text(pdf: Path) -> str:
    """Extract UTF-8 text from the canonical review PDF."""
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise SystemExit("missing required readability tool: pdftotext")
    result = subprocess.run(  # noqa: S603 - resolved fixed-name local tool
        [pdftotext, "-enc", "UTF-8", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_text(result.stdout)


def rounded(value: float) -> float:
    """Stabilize floating-point presentation."""
    return round(float(value), 4)


def build_report(text: str, source_name: str) -> dict[str, Any]:
    """Calculate readability metrics without applying arbitrary thresholds."""
    import nltk
    import textstat

    corpus_dir = Path(__file__).resolve().parents[1] / ".venv-arxiv" / "nltk_data"
    nltk.data.path.insert(0, str(corpus_dir))
    try:
        nltk.data.find("corpora/cmudict")
    except LookupError as error:
        raise SystemExit(
            "missing pinned Textstat corpus: sync the arXiv tool environment first"
        ) from error

    textstat.set_lang("en_US")
    words = int(textstat.lexicon_count(text, removepunct=True))
    sentences = int(textstat.sentence_count(text))
    if words < 50 or sentences < 3:
        raise SystemExit(
            "readability extraction is too small for review "
            f"({words} words, {sentences} sentences)"
        )

    warnings: list[str] = []
    smog: float | None = None
    if sentences >= 30:
        smog = float(textstat.smog_index(text))
    else:
        warnings.append(
            "SMOG omitted because the manuscript has fewer than 30 sentences"
        )

    return {
        "schema_version": "voiage.arxiv-readability.v1",
        "status": "review_only",
        "source": source_name,
        "language": "en_US",
        "textstat_version": version("textstat"),
        "interpretation": "Editorial evidence; no automatic acceptance threshold.",
        "counts": {
            "characters": int(textstat.char_count(text, ignore_spaces=True)),
            "difficult_words": int(textstat.difficult_words(text)),
            "polysyllable_words": int(textstat.polysyllabcount(text)),
            "sentences": sentences,
            "syllables": int(textstat.syllable_count(text)),
            "words": words,
        },
        "metrics": {
            "automated_readability_index": rounded(
                textstat.automated_readability_index(text)
            ),
            "coleman_liau_index": rounded(textstat.coleman_liau_index(text)),
            "dale_chall_readability_score": rounded(
                textstat.dale_chall_readability_score(text)
            ),
            "flesch_kincaid_grade": rounded(textstat.flesch_kincaid_grade(text)),
            "flesch_reading_ease": rounded(textstat.flesch_reading_ease(text)),
            "gunning_fog": rounded(textstat.gunning_fog(text)),
            "linsear_write_formula": rounded(textstat.linsear_write_formula(text)),
            "reading_time_seconds": rounded(textstat.reading_time(text)),
            "smog_index": rounded(smog) if smog is not None else None,
            "text_standard": textstat.text_standard(text),
        },
        "warnings": warnings,
    }


def main() -> None:
    """Extract a PDF and write its deterministic readability report."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_arxiv_readability.py INPUT.pdf OUTPUT.json")
    pdf = Path(sys.argv[1])
    output = Path(sys.argv[2])
    if not pdf.is_file():
        raise SystemExit(f"canonical review PDF does not exist: {pdf}")
    report = build_report(extract_text(pdf), pdf.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "readability evidence: pass "
        f"({report['counts']['words']} words, "
        f"{report['counts']['sentences']} sentences)"
    )


if __name__ == "__main__":
    main()
