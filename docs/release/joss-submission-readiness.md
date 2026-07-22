# JOSS and arXiv submission readiness

## Prepared files

- `paper/main.tex`: canonical authored LaTeX preprint.
- `paper/sections/`: semantic manuscript sections.
- `paper/references.bib`: bibliography packaged with the preprint.
- `paper/metadata.json`: reviewable arXiv metadata draft with category and
  license deliberately left as human decisions.
- `paper/readiness-manifest.json`: machine-readable evidence and human gates.
- `paper.md`: JOSS-format draft manuscript.
- `paper.bib`: BibTeX references.
- `codemeta.json`: software metadata for repository/archive workflows.
- `CITATION.cff`: release citation metadata.

## arXiv-first preparation

The repository uses the hardened `arxiv-paper-template` architecture as its
non-submitting arXiv readiness pipeline:

- `paper/main.tex` is the canonical source; Pandoc and Quarto are not part of
  the manuscript path;
- `scripts/build_arxiv_submission.py` compiles authored LaTeX, performs
  source/package hygiene checks, and writes a deterministic hashed tarball;
- SourceRight and Authentext are pinned submodules for source provenance and
  claim/evidence review;
- cleaner and collector variants are rebuilt, audited, and retained for human
  comparison rather than automatically selected;
- `.github/workflows/arxiv-readiness.yml` runs the reproducible preflight on
  manuscript changes, including TeX Live 2023/2025, ChkTeX, Lacheck, PDF/font
  assurance, and semantic LaTeXML HTML; it never uploads to arXiv.

Run `uv run python scripts/build_arxiv_submission.py` and review the generated
`build/arxiv/arxiv-readiness.json`, review PDF, source archive, checksum, and
variant diff before any future authenticated submission.

Optional cleaner and collector tooling is isolated from voiage's development
environment:

```console
uv venv .venv-arxiv
uv pip sync --python .venv-arxiv/bin/python requirements-arxiv.txt
.venv-arxiv/bin/python scripts/prepare_arxiv_variants.py
```

## Required author review before submission

- Confirm the author list, affiliations, ORCID identifiers, funding, and
  conflicts of interest.
- Replace the provisional research-impact statement with specific public
  publications, preprints, or research projects using `voiage`.
- Confirm the final citation title and release version.

## Preprint timing

JOSS’s current author policy explicitly permits submission to arXiv before,
during, or after JOSS submission and states that a preprint is not considered a
previous publication. Therefore, arXiv does not need to wait for JOSS. A
practical sequence is to finalize the author and impact evidence, submit the
arXiv version, then submit the same version to JOSS and link the arXiv record in
the JOSS form. Update both records when the JOSS DOI is assigned.

The paper is not submitted by repository automation. JOSS submission uses the
JOSS form, and arXiv submission requires the author’s arXiv account, category,
endorsement if applicable, and final confirmation.
