# arXiv source package

This directory defines the repository-owned arXiv preparation boundary for
`voiage`. The manuscript source is `paper.md`; this pipeline renders it with
Pandoc, retains the generated TeX, and packages only the
source, bibliography, metadata, and rendered PDF needed for review.

The workflow is intentionally a readiness gate. It does not upload to arXiv,
choose a subject category, certify authorship, or submit on the author's
behalf. Those are human gates.

## Local preflight

```console
uv run python scripts/build_arxiv_submission.py
```

The builder records `build/arxiv/arxiv-readiness.json`, checks that the
generated PDF and TeX source are reproducible inputs, scans the package for
secrets and machine-local paths, and writes `build/arxiv/voiage-arxiv.tar.gz`.
Use the repository's normal test and documentation commands before treating
the package as publication-ready.

The preferred compatibility target is arXiv's supported TeX Live release. A
local Pandoc/LaTeX build is evidence of reproducibility, not evidence of
submission or acceptance.
