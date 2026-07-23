---
name: arxiv-paper
description: Build and audit voiage's non-submitting arXiv source package.
---

# arXiv paper skill

Build with `uv run python scripts/build_arxiv_submission.py`. Inspect the canonical PDF, LaTeX
source archive, SHA-256 digest, semantic HTML, and readiness evidence. Generate
and inspect the review-only Textstat report from the canonical PDF, but do not
treat readability scores as scientific-quality thresholds. Never upload or
infer an arXiv identifier. Category, license, author, and final-upload decisions
remain human gates.
