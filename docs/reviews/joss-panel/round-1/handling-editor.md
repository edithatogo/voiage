# Handling-editor report

Reviewed revision: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`

Recommendation: major revision before submission

Score: **859/1000**

This is an internal simulated review, not a JOSS decision.

## Executive assessment

The manuscript is mechanically valid and broadly within JOSS scope. It has the
required sections, valid metadata, a word count within the required range, a
verified Software Heritage identifier, and a successful Open Journals build.
The central problem is the argument rather than the format.

The most defensible proposition is:

> `voiage` provides a validated decision-and-result contract for carrying
> labelled VOI analyses across language boundaries, while distinguishing
> shared numerical calculations from language-specific modelling and
> reporting.

That proposition should organise the paper. The current version does not yet
support it convincingly because:

- a synthetic example and an interoperability contract are both classified as
  research uses;
- the field comparison omits `dampack` and uses a regression-method paper as
  evidence for SAVI itself;
- the build-versus-contribute case speculates about other packages' purposes;
- the Software design section gives more attention to implementation mechanisms
  than their consequences for an applied VOI analysis;
- the release wording does not distinguish release assets, workflow
  attestations, and a separately retained software bill of materials.

## Recommended argument

1. A VOI result depends on formulas, strategy labels, units, uncertainty draws,
   population assumptions, and study-delivery assumptions.
2. Moving those elements between tools can change interpretation or require
   separate implementations.
3. Existing tools provide mature health-economic workflows but do not target
   the same language-neutral contract.
4. `voiage` represents those elements explicitly and shares selected
   calculations across languages.
5. A worked health example shows why population, timing, uptake, and study cost
   affect a research decision.
6. The repository has strong reviewer-facing software evidence, while external
   adoption remains unproven.

## Section architecture

| Section | Required revision |
| --- | --- |
| Summary | Lead with the decisions the software helps answer, reduce the method catalogue, and describe the language boundary in applied terms. |
| Statement of need | Name the labels, units, parameter groupings, population assumptions, and provenance that can be lost or altered. Separate user need from implementation. |
| State of the field | Add `dampack`, use a direct SAVI source, compare on explicit dimensions, and ground the separate-package decision in the project's requirements. |
| Software design | Explain information preservation and parity before implementation. Replace the generic “larger repository” trade-off with maintenance and parity costs. Condense the assurance catalogue. |
| Research impact | Separate one research demonstration from one interoperability demonstration and report a restrained result from the health example. |
| AI usage disclosure | Tighten only if every required disclosure remains. |
| Acknowledgements | Retain the funding, sponsor-involvement, and competing-interest declarations; remove the generic acknowledgement unless specific contributions are named. |
| References | Move release, generated-data, and archive information to a short Software and data availability section. |

## Claim and citation findings

Supported:

- the foundational VOI sources;
- the current Rust, Python, R, and Julia boundary;
- the existence of the stated classes of tests and workflows;
- the Software Heritage snapshot;
- the fixed-seed health example.

Required corrections:

1. Lines 118–126 overclassify the synthetic example and the same-author
   `vop_poc_nz` contract as research uses.
2. Lines 76–78 need a direct source for SAVI itself.
3. The comparison should include `dampack`.
4. Lines 86–88 should explain the author's separate-package choice without
   asserting another package's intended purpose.
5. Lines 110–114 should distinguish the visible release assets from
   attestations and separately retained SBOM evidence.
6. The `voi` package year should be reconciled: the package version and the DOI
   registration year are not currently distinguished.
7. The software-citation sentence at lines 159–160 is peripheral and can be
   removed unless metadata is part of the argument.

## Sentence-level priorities

- Lines 30–39: define VOI in terms of the research decisions it informs and
  reduce the API-like method list.
- Lines 41–48: remove unexplained “binding-independent”, “domain contracts”,
  and “serialization” language from the Summary.
- Lines 52–58: replace generic “fragmentation” with the specific information
  that has to survive transfer.
- Lines 60–68: replace “clear” and “careful” self-evaluations with observable
  behaviour; separate audience from architecture.
- Lines 74–78: complete and directly support the field comparison.
- Lines 82–90: state the unmet requirement and the resulting trade-off without
  attributing intentions to other maintainers.
- Lines 95–100: replace PyO3 and C-interface detail with reader-facing
  consequences; retain low-level detail in the documentation.
- Lines 102–108: describe maintenance and parity costs; express maturity
  limits in plain language.
- Lines 110–114: replace the test catalogue with a shorter failure-mode account
  and accurately locate release evidence.
- Lines 118–131: distinguish research, demonstration, and interoperability;
  report a health-example result; preserve the absent-adoption boundary.
- Lines 135–146: retain the complete AI disclosure with lighter list cadence.
- Lines 150–153: remove the generic thanks and add sponsor-involvement wording.
- Lines 157–160: move availability facts and remove the peripheral
  software-citation assertion.

## Rubric

| Dimension | Score |
| --- | ---: |
| Scope, significance, and research use | 150/180 |
| Statement of need and audience | 110/120 |
| State of the field and build-versus-contribute case | 100/130 |
| Scientific and numerical accuracy | 145/150 |
| Software design and research relevance | 85/100 |
| Reproducibility, packaging, documentation, and tests | 90/100 |
| Research-impact statement | 55/80 |
| Structure, metadata, and JOSS format | 58/60 |
| Clarity, accessibility, and sentence quality | 48/55 |
| Citations, provenance, declarations, and AI disclosure | 18/25 |
| **Total** | **859/1000** |

## Manuscript blockers

1. The impact statement overstates its evidence.
2. The field comparison and build-versus-contribute case are incomplete.
3. The SAVI software claim lacks a direct source.
4. The release and SBOM wording is not accurately scoped.
5. Sentence-level revision remains necessary across the Summary, Statement of
   need, Software design, and Research impact statement.

## External gates

The licence, public history, iterative development, tests, documentation,
mechanical validator, and official build are ready at the reviewed revision.
Issue #471 has no non-author report. The permanent arXiv identifier is
externally pending but is not a JOSS requirement. Submission, review, the
review-version DOI archive, and acceptance remain unperformed external steps.
