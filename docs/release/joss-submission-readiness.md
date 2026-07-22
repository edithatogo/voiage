# JOSS and arXiv submission readiness

## Prepared files

- `paper.md`: JOSS-format draft manuscript.
- `paper.bib`: BibTeX references.
- `codemeta.json`: software metadata for repository/archive workflows.
- `CITATION.cff`: release citation metadata.

## arXiv-first preparation

The repository now has a separate, non-submitting arXiv readiness pipeline:

- `paper/arxiv/README.md` documents the package boundary and human gates;
- `scripts/build_arxiv_submission.py` renders `paper.md` to TeX and PDF,
  performs source/package hygiene checks, and writes a hashed tarball;
- `.github/workflows/arxiv-readiness.yml` runs the reproducible preflight on
  manuscript changes and never uploads to arXiv.

Run `uv run python scripts/build_arxiv_submission.py` and review the generated
`build/arxiv/arxiv-readiness.json` before any future authenticated submission.

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
