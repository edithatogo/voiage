# Current pre-submission requirements baseline — 2026-08-29

This baseline interprets the primary sources pinned in
`current-requirements-source-manifest-20260829.json`. It is an input to the
whole-product gap analysis, not a claim of compliance, submission, review, or
acceptance.

## Route and authority boundary

All repository repair and assurance comes first. The selected external route
remains pyOpenSci, then the JOSS partner fast track, then a separate rOpenSci
review. Each action is sequential because both pyOpenSci and rOpenSci warn
against concurrent review of the package or associated manuscript. pyOpenSci
acceptance can replace a second JOSS software review only if JOSS independently
finds the package in scope; JOSS still reviews the paper. rOpenSci review
remains a full R-package and statistical-software review and receives no
automatic exemption from either earlier outcome.

No inquiry, submission, survey, reviewer communication, upload, release,
acceptance, DOI, badge, transfer, or registry action is authorized by this
baseline.

## Normative venue criteria

### pyOpenSci

- Scientific-workflow fit, ecosystem comparison, a stable or nearly stable
  public API, installable packaging, comprehensive user-facing documentation,
  examples, tests, CI, metadata, contribution/support paths, and an
  OSI-compatible licence are reviewable requirements.
- The public history must demonstrate sustained, iterative human-led
  development rather than a recent bulk generation event. Current policy asks
  submitters to attest to roughly 3–6 months of public development history.
- Generative-AI use must be disclosed and all submitted code and documentation
  must be understood and carefully reviewed by humans. Review-issue
  communication is expected to be written by a human, apart from restrained
  translation or grammar assistance.
- Maintainers must intend to support the accepted package for at least two
  years, respond to review, and provide succession or archival handling if
  maintenance ends. Inactivity monitoring and a best-effort response policy
  must not be represented as a guaranteed service level.
- The package must be mature enough to review in full. pyOpenSci does not
  supply scientific validation for an unreviewed novel method; method and
  evidence claims therefore remain bounded by existing scientific evidence.

Sources: `PYOS-POLICY`, `PYOS-AUTHOR`, `PYOS-REVIEWER`, `PYOS-SCOPE`, and
`PYOS-JOSS`.

### JOSS

- The repository must be openly browsable and contribution-ready, use an
  OSI-approved licence, have an obvious research application, and contain
  feature-complete, maintainable, appropriately packaged software rather than
  a one-off analysis or thin utility.
- Pre-review screening requires more than six months of public, iterative
  development, good open-source practice, and demonstrated research use or
  credible external integration. Aspirational future impact is insufficient.
- Review covers installation, functional claims, performance claims, tests,
  documentation, examples, contribution/reporting/support paths, authorship,
  and community evidence. Performance claims require reproducible evidence.
- The JOSS paper must satisfy the current Markdown/YAML format and 750–1750
  word target, including Summary, Statement of need, State of the field,
  Software design, Research impact statement, AI usage disclosure,
  acknowledgements, and complete references. The repository's canonical LaTeX
  preprint remains separate; a validated `paper.md` handoff must not replace
  `paper/main.tex`.

Sources: `JOSS-SUBMIT`, `JOSS-CHECKLIST`, `JOSS-CRITERIA`, and `JOSS-PAPER`.

### rOpenSci statistical software review

- The R package must be in scope, mature, fully documented and tested, have a
  stable essential API, follow the rOpenSci packaging guide, and be maintainable
  for at least two years. The README must establish purpose, use, overlap, and
  package scope without requiring installation.
- A locally reproducible current `pkgcheck` run is required. Exceptions must be
  explicitly justified; successful CRAN-style checks alone are not equivalent
  to rOpenSci readiness.
- `roxygen2`-generated documentation and namespace, at least one vignette,
  usable examples, metadata/authorship, dependency hygiene, CI, and test
  coverage are explicit automated-review concerns.
- Statistical packages must use `srr` and document every applicable general
  and category-specific standard with item-level `@srrstats` evidence or a
  justified `@srrstatsNA`, then pass `srr_stats_pre_submit()`.
- The standalone source package must install and run without an undeclared
  external native library. R CMD check, examples, vignettes, installed-runtime
  numerical fixtures, and supported-platform evidence remain separate gates.

Sources: `ROSCI-POLICY`, `ROSCI-AUTHOR`, `ROSCI-PACKAGING`,
`ROSCI-PKGCHECK`, `ROSCI-SRR`, and `ROSCI-STANDARDS`.

## Packaging, security, and research-software baseline

- Python distributions use declarative `pyproject.toml` metadata and build
  isolated sdist/wheel artifacts. CI should build once, retain immutable
  artifacts, test the artifacts, and publish those exact artifacts in separate
  jobs through PyPI Trusted Publishing rather than long-lived upload tokens.
- Binary-extension coverage is assessed across supported interpreters and
  platforms; the source distribution must be able to recreate wheels with
  declared build dependencies.
- SLSA 1.2 is the current approved supply-chain specification. Provenance must
  identify subjects, source and build inputs, builder, and invocation. The
  repository must claim only the level actually evidenced; two-party source
  review is not assumed for a solo-maintainer project.
- OpenSSF Scorecard is an advisory diagnostic, not a release or quality score.
  Findings for branch protection, token permissions, pinned dependencies,
  SAST, vulnerabilities, packaging, signed releases, and maintenance require
  evidence-based disposition; solo-maintainer code-review scoring is treated
  as context rather than falsified compliance.
- FAIR4RS findability, accessibility, interoperability, and reusability are
  evaluated through persistent metadata, citation, licence, releases,
  standards, portable formats, documentation, provenance, and reproducible
  examples.

Sources: `PYPA-FLOW`, `PYPA-TOOLS`, `PYPA-PUBLISH`, `SLSA-1.2`,
`OSSF-SCORECARD`, and `FAIR4RS-1.0`.

## Mandatory downstream checks

The Phase 1 gap analysis must test every criterion above against source and
installed artifacts. Any unmet criterion is dispositioned later as must-fix,
accepted limitation, preview-only capability, reviewed exclusion, external
gate, or human gate. A green test, generated checklist, badge score, or local
packet cannot manufacture human attestation or external acceptance.
