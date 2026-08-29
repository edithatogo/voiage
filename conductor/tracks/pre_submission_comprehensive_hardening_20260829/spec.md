# Pre-Submission Comprehensive Hardening

## Overview

Create one authoritative, evidence-gated programme for all repository-owned
work that should be completed before `voiage` is submitted to pyOpenSci,
rOpenSci, or JOSS. The programme consolidates every active Conductor track,
refreshes the whole-product gap analysis, repairs accepted gaps, and records
reviewed exclusions without turning external or human evidence into a local
completion claim.

## Requirements

1. Consolidate every active Conductor track into this track. Preserve original
   specifications, plans, completed-task evidence, GitHub references, and
   append-only ledgers in the archive. Every migrated pending task must retain
   its source track and task identifier in a migration manifest.
2. Refresh the bounded feature and ecosystem analysis against current primary
   sources. Cover core and frontier VOI methods, estimands, estimators,
   diagnostics, integrations, data workflows, worked examples, competing
   software, and credible adjacent research-software expectations.
3. Audit architecture and maintainability across the Rust numerical authority,
   Python facade, C ABI, R and Julia bindings, CLI, schemas, serialization,
   documentation, packaging, and release boundaries.
4. Audit every public API and retained ABI for consistency, compatibility,
   error behavior, versioning, capability discovery, installed-package
   behavior, and honest maturity labels.
5. Make `voiageR` independently installable, resolve the external-library
   contradiction in its package metadata, implement item-level rOpenSci `srr`
   annotations or justified exclusions, and pass `pkgcheck`, coverage, and
   multi-platform `R CMD check` evidence.
6. Refresh stable and preview dependency frontiers using `uv lock --upgrade`
   and `scripts/dependency_frontier.py . --strict`. Evaluate improved,
   bleeding-edge, experimental, and preview dependencies only in named,
   isolated lanes with compatibility, numerical-equivalence, Arrow
   round-trip, CPU-fallback, security, and reproducibility evidence.
7. Profile the test and build system before optimizing it. Use Scalene and
   native timing data to identify Python CPU, memory, import, collection, I/O,
   and serialization costs; use Rust, R, Julia, docs, and workflow-native
   timing where Scalene is not authoritative.
8. Improve CI/CD and local testing through evidence-backed parallelism,
   dependency and build caching, deterministic test sharding, change-aware
   focused lanes, reusable artifacts, cancellation/concurrency controls, and
   removal of duplicate setup or validation. Preserve a full fail-closed
   release gate and do not trade correctness or reproducibility for elapsed
   time.
9. Reconcile pyOpenSci, rOpenSci, and JOSS requirements against their current
   official guidance. Bind submission materials, manuscript claims, release
   evidence, and installed-package validation to one exact final revision and
   version.
10. Classify every finding as `must_fix`, `accepted_limitation`,
    `experimental_or_preview`, `reviewed_exclusion`, `external_gate`, or
    `human_gate`. Only `must_fix` repository work blocks repository readiness;
    all other dispositions require explicit evidence and accurate public
    documentation.
11. Finish with independent automated review, the complete repository harness,
    full tox validation, language-native checks, documentation builds,
    security and dependency audits, and exact-head hosted required checks.
12. Do not perform any authenticated venue, registry, archive, badge, funding,
    or publication submission in this track.

## Acceptance criteria

- **AC-01:** The migration manifest accounts for all 21 source tracks and every
  pending or in-progress task; the source tracks are archived as superseded
  without rewriting their completed evidence.
- **AC-02:** A dated, source-linked feature and ecosystem gap report covers
  analytical, integration, data, dependency, CI/CD, architecture, API, ABI,
  packaging, documentation, governance, and research-software dimensions.
- **AC-03:** Every finding has an evidence-backed disposition, owner boundary,
  validation path, and release/submission impact.
- **AC-04:** All accepted `must_fix` repository findings are implemented,
  tested, documented, and independently reviewed; accepted limitations and
  exclusions are visible in user-facing capability surfaces.
- **AC-05:** Stable Python behavior is Rust-authoritative where promised; API,
  C ABI, R, and Julia contracts are versioned and installed-artifact tested.
- **AC-06:** `voiageR` is standalone and passes item-level `srr`, `pkgcheck`,
  coverage, examples, vignettes, and multi-platform `R CMD check` gates.
- **AC-07:** Dependency refresh and preview-lane decisions are backed by
  compatibility, numerical, security, provenance, and fallback evidence.
- **AC-08:** A baseline and optimized CI/test performance report demonstrates
  improvements without weakened coverage, skipped required gates, flaky
  sharding, or hidden serial bottlenecks.
- **AC-09:** pyOpenSci, rOpenSci, and JOSS readiness artifacts match current
  official requirements and the exact final candidate release.
- **AC-10:** Full local and exact-head hosted gates pass, the working tree is
  clean, and submissions remain explicitly unperformed.

## Non-functional constraints

- Preserve deterministic numerical results, provenance, rights, privacy,
  backward compatibility, and fail-closed maturity boundaries.
- Optimize measured bottlenecks only. Scalene evidence is diagnostic and may
  not become a universal timing oracle for non-Python work.
- Preview features remain isolated and non-blocking until their promotion
  criteria pass; stable users must retain a CPU-supported path.
- Do not manufacture community adoption, scientific review, contributor
  diversity, rights clearance, or human attestations.
- Keep pull requests as the auditable change boundary and preserve the
  repository's solo-maintainer automated-review policy.

## External and human gates

- pyOpenSci attestations, survey, posting, review, and acceptance.
- Genuine non-author community engagement and JOSS editorial/reviewer work.
- rOpenSci pre-submission inquiry, editorial category decision, review, and
  acceptance.
- arXiv category/licence selection and authenticated upload.
- Rights or security approval for live/remote data access.
- Independent domain or scientific judgments that cannot be established by
  repository tests.
- Registry, badge, fiscal-host, funding, archive, DOI, and publication actions.

## Out of scope

- Performing any external submission or claiming an external outcome.
- Promoting every experimental method to stable solely to increase a feature
  count.
- Requiring every language binding to expose every experimental Python method.
- Replacing the canonical LaTeX preprint with the JOSS manuscript.
- Buying infrastructure or enabling paid runners without separate authority.

## Authoritative inputs

- `AGENTS.md`, `roadmap.md`, `todo.md`, `CONTRIBUTING.md`
- `conductor/product.md`, `conductor/product-guidelines.md`,
  `conductor/tech-stack.md`, `conductor/workflow.md`
- `specs/`, `voiage/`, `rust/`, `r-package/`, `bindings/`, `tests/`, `tox.ini`,
  `pyproject.toml`, `uv.lock`, and `.github/workflows/`
- pyOpenSci author guide and policies, observed 2026-08-29:
  <https://www.pyopensci.org/software-peer-review/how-to/author-guide.html>
- rOpenSci author and statistical software guides, observed 2026-08-29:
  <https://devguide.ropensci.org/softwarereview_author.html> and
  <https://stats-devguide.ropensci.org/>
- JOSS submission and review criteria, observed 2026-08-29:
  <https://joss.readthedocs.io/en/latest/submitting.html> and
  <https://joss.readthedocs.io/en/latest/review_criteria.html>
