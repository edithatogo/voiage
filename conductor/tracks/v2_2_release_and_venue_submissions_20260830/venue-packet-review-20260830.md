# R13 venue-packet repair review — 30 August 2026

## Scope and findings

The public release remains v2.2.0 at `7af563c8`; these later packet repairs do
not retag or replace its distributions. No venue submission, reviewer message,
survey, author certification, archive DOI, or acceptance was performed.

1. Corrected both manuscript sources and claim evidence: Python and Julia use
   the main Rust core; R bundles a separate offline Rust kernel. Both narrower
   bindings expose native EVPI and ENBS, not Python's full decision records.
2. Preserved historical v2.0.0 research use and Software Heritage evidence.
   R13a's supported-environment v2.2.0 replay is automated computational
   evidence, not new human use or independent adoption.
3. Corrected AI-transparency projections. The July human attestation remains
   historical; autonomous delivery authorization does not certify personal
   review of later work. Exact missing model identifiers and generated-code
   percentages are not invented.
4. Retained the unchanged official pyOpenSci template revision and added its
   current supplemental policy requirements. The selected JOSS option is
   confirmed; personal code understanding, human-led history, disclosure scope,
   communication and other form declarations remain unchecked and fail closed.
5. Bound the rOpenSci packet to the checked 2.2.0 source archive and unchanged
   R subtree. The first CRAN-incoming timeout and the successful retry's
   unavailable remote incoming checks remain explicit, not passed claims.
6. PDF inspection found and repaired a stale version-1 table caption. Local
   LaTeXML returned zero despite omitting its figure because the Perl image
   module is missing. The validator now rejects that incomplete artifact;
   hosted complete semantic HTML is required before merge.
7. Submission preflight wrongly treated `pass` and `complete_internal_review`
   as incomplete and allowed missing gate dictionaries to pass vacuously.
   It now checks required gate names, recognizes their completed statuses,
   checks current authorship declarations and explicitly enforces the selected
   partner route. Eight initially failing regression cases now pass; hypothetical
   test acceptance never changes the real pending venue evidence.

## Local manuscript evidence

- JOSS source SHA-256:
  `fc8a9eb77906b8657385d397acff717fa7b9eac738d9d28abcb3505331ee787a`.
- JOSS bibliography SHA-256:
  `03906c30c062037e6af183d6891a21cd361ffe369f4691b37f8bf6880aa5ebe7`.
- Article contract: 1,625 body words, all section bands passed. The synthetic
  worked-example files match clean regeneration; scientific results unchanged.
- SourceRight: 18 citation occurrences matched, zero structural issues, six
  software/web no-DOI warnings and 15 queued reference checks. This is not
  source-truth certification. The retained audit manifest is
  `venue-sourceright-audit-20260830.json`.
- Authentext: zero findings in its selected deterministic pattern set; retained
  as `venue-authentext-audit-20260830.json`, not an exhaustive human review.
- Canonical preprint PDF: 15 pages, 26 embedded fonts; all 15 rendered pages
  inspected, including both tables, equations, figure, links and references.
  No clipping, overlap, missing glyphs or missing figures observed.
  PDF SHA-256:
  `17c35bdd7abc9fd99bb6e1828de33304bf88649d8a8db9ef94b079ef744ee1ab`.
- Deterministic source archive: two builds byte-identical; SHA-256
  `40b8abcb9c81514c790a72a19165bacbc307305e473b183a48168be902e22edb`.
- Textstat: 4,267 words, 293 sentences; review-only metrics retained in
  `preprint-readability-20260830.json`, without an acceptance threshold.
- Prior local preprint/variant builds were moved to recoverable ignored
  `.conductor/local/` directories before regeneration.
- Cleaner and collector variants compile, pass source/PDF audits, and produce
  identical extracted text to the canonical PDF. The cleaner diff removes a
  source comment and unused ancillary files, not scientific content. No variant
  was selected for upload. The participant protocol now regenerates into a
  temporary directory before comparing portable CSVs, preserving retained
  figure checksums without claiming cross-platform rendering identity.

Final full tox passed all 15 environments: 4,587 tests passed, 16 skipped,
95.16 percent coverage, 402.99 seconds. The log SHA-256 is
`5e40fdcd6b5546e51ec9d281459676bf93e18423d1dbdb004d036da01447bae9`.
The HTML validator has 10 passing focused tests and 96.77 percent script
coverage; 28 JOSS/preflight tests and 32 pyOpenSci staging tests passed.
Direct Ruff/type checks, full Conductor/cross-reference validation, submission
contracts, Vale and whitespace checks passed. The handoff script's separate
VOP imports are excluded from unresolved-import checking in the voiage-only
environment; its actual two-environment execution is recorded separately.
Final current hosted artifact review remains required before merge.
The local HTML preview is deliberately not certified as complete.

The first full run passed 14 environments but failed one historical registry
test that required the July human attestation to certify the current manuscript
(4,577 passed, 16 skipped; coverage 95.16 percent). The test now checks the
dated historical attestation separately from current pending review. No human
confirmation was fabricated to make it pass. A fresh complete tox run follows
that correction and the late submission-preflight regression repair.

## Current external boundaries

Checked current primary policies:
[pyOpenSci](https://www.pyopensci.org/software-peer-review/our-process/policies.html),
[JOSS submission](https://joss.readthedocs.io/en/latest/submitting.html),
[JOSS paper](https://joss.readthedocs.io/en/latest/paper.html), and
[rOpenSci author guide](https://devguide.ropensci.org/softwarereview_author.html).

- pyOpenSci issues #271 and #272 remain open; an on-hold label is not assumed
  to exempt the contact from the one-review-at-a-time policy. Eligibility needs
  human/editorial resolution, not another agent-invented contact.
- Review communication must be human-led; JOSS prohibits AI assistance in
  author/editor/reviewer conversations except translation. The local staging
  draft is not a message sent to a venue.
- Issue #471 still has only author comments. Genuine non-author engagement,
  the author-selected arXiv sequence, reviewed-version archive and eligible
  pyOpenSci-to-JOSS handoff remain separate gates.
- Final source/claim review and current AI disclosure confirmation, personal
  form declarations and survey, venue scope/maturity decisions, and external
  editorial outcomes cannot be completed by automated checks.
- No concurrent venue review is initiated. The track and #1037 stay open.
