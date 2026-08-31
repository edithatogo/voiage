# Additional venue assessment

Repository assessment on 31 August 2026 for #615 and #614: **defer a separate
R Journal or JSS manuscript**. Preserve the authorized pyOpenSci-first and
eligible JOSS route. This records a preparation recommendation, not an author
attestation or editorial decision, and creates no duplicate manuscript.

## Evidence and distinct contribution

The current standalone R surface builds its bundled Rust EVPI and ENBS kernels
offline. Its optional EVPPI and EVSI paths delegate to Python. These are useful
installation and language-access improvements, but the current evidence does
not establish a distinct R-centred research contribution beyond the software
already described in `paper.md`. A second language wrapper alone is not the
substantial new article proposed by the roadmap.

The applicable standards mapping is general plus empirical Probability
Distributions, not a Bayesian sampler implementation. Preserve that reasoned
classification in `specs/submission-readiness/ropensci-standards-mapping.md`;
do not add nominal Bayesian/Monte Carlo annotations merely to satisfy older
issue wording. The v2.2 source-check receipt also retains a time-verification
NOTE and unavailable remote CRAN incoming checks. It does not establish the
strict zero-error, zero-warning, zero-note outcome requested by #1024.
A fresh check of the same immutable archive on 31 August reached CRAN incoming
feasibility and completed with zero errors, zero warnings and two notes:
"New submission" and "unable to verify current time". No check was suppressed;
the manual was excluded as in the prior recorded command. The dated receipt
in the remaining-backlog track preserves this partial result.

| Candidate | Distinct outcome required | Current assessment and next trigger |
| --- | --- | --- |
| R Journal | A reproducible contribution of interest to R users, beyond repackaging the same software description | Defer until the R package and an R-specific worked evaluation provide a demonstrably separate article; recheck distribution and submission requirements then |
| JSS | A substantial statistical-software account with reproducible results beyond the short JOSS software paper | Defer until the methods, evaluation and overlap audit demonstrate the additional contribution; do not assume a longer rewrite is enough |
| NumFOCUS affiliation | Community and sustainability support rather than another paper | Prepare locally; current applications are paused and community/governance evidence remains incomplete |
| Zenodo | A selected DOI-bearing archival purpose distinct from retained release and Software Heritage evidence | Defer until that need and deposition authority are established; no identifier is invented |

## Authoritative sources and boundaries

The [R Journal submission guidance](https://rjournal.github.io/submissions.html)
requires reproducible material and excludes articles already published or
submitted elsewhere. Its checks include package availability through CRAN or
Bioconductor. The [JSS preparation checklist](https://www.jstatsoft.org/about/submissions)
requires a manuscript, software source and replication materials. Neither page
provides automatic acceptance or pre-approval for the proposed voiage articles.
The [NumFOCUS status page](https://numfocus.org/projects-overview) currently
reports paused applications. Recheck policies before any future submission.

Repository evidence reviewed: `r-package/voiageR/README.md`, `paper.md`,
`specs/submission-readiness/ropensci-evidence.json`,
`specs/submission-readiness/r-v2-2-archive-check-20260830.json` and the standards
mapping. The deferred recommendations do not mark pending distinct-contribution
requirements satisfied. No changes were made to canonical `paper/main.tex`.

## Full R check follow-up, 1 September 2026

The same checksum-bound archive passed a full `R CMD check --as-cran` with
PDF and HTML manual checks enabled. It returned zero errors, zero warnings
and the same two NOTEs: new submission and unavailable current-time
verification. No check was suppressed and no source was edited. The manual
verification gap is resolved; the strict zero-NOTE criterion remains unmet.
See `conductor/tracks/remaining_backlog_delivery_20260831/r-manual-check-20260901.json`
for the command, archive, log and manual hashes. No CRAN submission occurred.
