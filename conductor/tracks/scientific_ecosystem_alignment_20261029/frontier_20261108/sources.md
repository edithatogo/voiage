# Primary sources and inspection boundaries

Checked: 2026-11-08. These links establish candidate resources or current guidance,
not compatibility, numerical correctness, redistribution permission or acceptance.
Pin exact source bytes/revisions during Q03/Q09 admission. This planning pass did
not download or execute the external benchmark datasets.

## Public sources checked

- **S01 — ConVOI `voi`:** [reference index](https://chjackson.github.io/voi/reference/index.html) and [EVSI contract](https://chjackson.github.io/voi/reference/evsi.html). Includes chemotherapy model objects/functions and decision/estimation VOI interfaces. Use as a comparator and source of reproducible problem definitions, not as unquestionable ground truth.
- **S02 — BCEA:** [reference index](https://n8thangreen.github.io/BCEA/reference/index.html). Cost-effectiveness/VOI functions and example datasets; verify the selected data objects, versions and licences before export.
- **S03 — DARTH:** [Sick–Sicker introduction](https://darth-git.github.io/darthpack/articles/aa-introduction.html). Reproducible decision-model and calibration candidate; independently check model conventions rather than copying narrative numbers without review.
- **S04 — gemtc smoking candidate:** [CRAN source manifest](https://github.com/cran/gemtc/blob/master/MD5). The manifest lists `data/smoking.rdata`. This is a source-location observation only; source bytes, licence and an economic decision model remain admission work.
- **S05 — sbibm:** [upstream benchmark](https://github.com/sbi-benchmark/sbibm). Reference inference tasks must be augmented with decision actions and utility; posterior similarity alone cannot verify VOI. Check which tasks have usable reference posteriors.
- **S06 — Uptake data:** [OWID vaccination series and source notes](https://ourworldindata.org/grapher/share-people-fully-vaccinated-covid). Candidate forecast-validation series; third-party source rights require separate checking. It does not identify causal implementation effects.
- **S07 — JOSS:** [submission requirements and AI policy](https://joss.readthedocs.io/en/latest/submitting.html). Confirm scope, research use, packaging, public history and current author/reviewer interaction rules before submission.
- **S08 — pyOpenSci:** [author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html) and [scope](https://www.pyopensci.org/software-peer-review/about/package-scope.html). The JOSS route is conditional on eligibility, not guaranteed publication. Check operational intake at submission rather than relying on an old pause notice.
- **S09 — Medical Decision Making:** [journal scope](https://smdm.org/medical-decision-making-journal/), [manuscript requirements](https://www.journals.smdm.org/manuscript-requirements/), and [submission policy](https://www.journals.smdm.org/manuscript-submission/). Recommended evaluation destination; acceptance is external.
- **S10 — Value in Health:** [author guidance](https://www.ispor.org/publications/journals/value-in-health/for-authors/guide-for-authors). Relevant to substantive HEOR methodology; refresh article type and disclosures before submission.
- **S11 — JSS:** [mission and scope](https://www.jstatsoft.org/mission). Reproducible statistical-software contribution; distinguish software innovation from methodological innovation and avoid duplicate publication.

Statistics in Medicine and SIAM/ASA UQ are candidate destinations requiring a fresh
paper-specific fit/policy check, not verified submission-ready routes in this pass.
Survival, prediction-model and causal/RWE reproductions need a new primary-literature
search and explicit code/data admission; no unverified bibliographic claims from
prior conversation are promoted into the evidence record.

## Repository baseline inspected

Baseline commit: `a5d87a736fa9f0f876e44872d946b2170d2a0898`.
Relevant records: `AGENTS.md`; current priority sections of `roadmap.md` and `todo.md`;
`CONTRIBUTING.md`; the quality/security guide; `conductor/index.md`; this parent
track's index/plan; the existing agent-safe `repository-boundaries.md`; and the
R package README. The source baseline is inspection evidence, not a test result.
Earlier assessment findings about duplicated arithmetic, matrix copies and broader
method semantics must be reconfirmed against source and executable witnesses in Q01/Q02.

No independent live census of every sibling repository or Hub asset is claimed.
The existing ownership map supplies candidate responsibilities; Q08 requires fresh
source, licence and installed-consumer evidence before admission. The proposed
`edithatogo/voiage-benchmarks` resource is not asserted to exist.

## Verification boundary for this planning PR

The editing environment has Python 3.13.5, not the required repository Python
3.15.1. A direct Git clone failed because the runtime could not resolve GitHub;
repository inspection and writes therefore use the connected GitHub tools.
Only local planning-manifest/static consistency checks and remote file/diff
verification are eligible evidence here. The full tox/Cargo/binding/benchmark
suites are not run or claimed passing. Keep the PR draft until required repository
verification and review are supplied; do not change or bypass those gates.
