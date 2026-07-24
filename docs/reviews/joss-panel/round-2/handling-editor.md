# Handling-editor report: round 2

Reviewed repository state: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`
plus the uncommitted shared-worktree changes present on 24 July 2026.

Recommendation: **major revision before submission**

Score: **918/1000**

Fail-closed status: **manuscript/release provenance blocker present; score capped
below 950**

This is an internal AI-assisted simulation, not a JOSS editorial decision.

## Executive assessment

The second-round manuscript is substantially stronger. It is concise,
non-promotional, understandable outside software engineering, and organised
around the decisions VOI informs. It now:

- distinguishes the broad Python interface from the EVPI-only R and Julia
  interfaces;
- compares `voiage` with `voi`, `BCEA`, `dampack`, and SAVI;
- gives a defensible build-versus-contribute argument without attributing
  motives to other package maintainers;
- reports the worked health-example results;
- describes the examples as author-created demonstrations rather than
  independent adoption;
- corrects the `voi` authorship record;
- distinguishes GitHub release assets and attestations from the separately
  generated SBOM; and
- provides complete funding, conflict-of-interest, authorship, and AI-use
  statements.

The central manuscript argument is now coherent:

> VOI analyses depend on labels, units, assumptions, population and
> implementation settings, and provenance as well as numerical arrays.
> `voiage` preserves that decision description while sharing selected Rust
> calculations across language interfaces whose different capability levels
> are stated explicitly.

The remaining material defect is version identity. The manuscript describes a
Rust normal--normal EVSI implementation and corrected R native path that exist
in the current uncommitted worktree, but release `v1.0.0` predates both. The
availability section then points to `v1.0.0` and says that the exact reviewed
revision “will be frozen before submission.” A JOSS reader cannot reproduce
the software described by installing the cited release, and the reviewed
software state is not an immutable commit. This is an unresolved round-one
finding and a fail-closed provenance defect.

The absence of attributable non-author use is stated accurately and is not a
manuscript misrepresentation. It remains a material JOSS pre-review risk under
issue #471. More importantly, the current paper describes demonstrations but
does not identify a completed research analysis in which `voiage` was used.
Current JOSS screening requires demonstrated research use at least by the
developers. That eligibility question needs evidence outside copy-editing.

## Evidence inspected

- the complete current `paper.md` and `paper.bib`;
- every round-one panel report and the round-one synthesis;
- the current Rust analytical normal--normal EVSI implementation, Python
  exposure, compatibility-estimator warnings, and reference tests;
- the current R and Julia EVPI bindings and their installation boundaries;
- the fixed-seed health-example CSV outputs;
- the governed method-maturity contract;
- the public `v1.0.0` tag and GitHub release assets;
- Software Heritage and release-evidence records;
- issue #471 and its independent-validation protocol;
- the current JOSS submission, paper, review, and editorial guidance; and
- the repository-owned JOSS validator and tests.

The repository-owned validator passed, and all seven focused JOSS-readiness
tests passed. The current paper body remains within the 750--1,750-word range.
An official Open Journals render was demonstrated in round 1, but no immutable
commit or hosted Open Journals build exists for this uncommitted round-two
state.

## Round-one disposition

| Round-one finding | Round-two status | Handling-editor assessment |
| --- | --- | --- |
| Author-created demonstrations were called research uses | Resolved in the manuscript | Lines 117--135 use “demonstration” and expressly deny independent adoption. |
| `dampack` was omitted | Resolved | Lines 69--76 include an accurate, directly cited comparison. |
| SAVI was supported only indirectly | Resolved | A direct SAVI record is now paired with the Strong et al. method citation. |
| Build-versus-contribute reasoning speculated about other maintainers' intentions | Resolved | Lines 78--84 state the project's requirement and maintenance trade-off. |
| Cross-language claims exceeded the common implementation | Resolved in wording | Lines 41--45 and 88--95 limit R and Julia to EVPI and acknowledge their narrower interfaces. |
| The public EVSI claim exceeded validated evidence | Resolved in the current worktree, not in a release | A declared Rust normal--normal model and corrected Python path now exist, while compatibility estimators are non-stable. The cited `v1.0.0` release does not contain this work. |
| The installed R native EVPI path failed | Resolved locally, not release-bound | The symbol-address fix and native test exist in the worktree. R and Julia still require a separately built native library. |
| Release wording implied that the SBOM was a release asset | Resolved | Lines 110--113 distinguish release assets, attestations, and a separate SBOM workflow. |
| `voi` authorship metadata was wrong | Resolved | The bibliography lists Christopher Jackson and Anna Heath as authors. |
| Health-example results were absent | Resolved | Lines 120--128 report the fixed-seed results and match the current generated CSVs. |
| Availability did not identify the exact reviewed revision | **Unresolved blocker** | Lines 161--162 retain a future-tense freeze promise instead of an immutable revision and matching release. |
| Attributable non-author evidence was absent | Externally unresolved | The boundary is honest; issue #471 contains only the author's request. |

## Rubric score

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 163/180 | Clear research-software scope and plausible usefulness, but no completed research analysis or independent adoption is identified. |
| Statement of need and audience | 118/120 | The need and audience are specific; the cross-domain sentence is broader than the evidence presented. |
| State of the field and build-versus-contribute case | 124/130 | The principal health-economic alternatives are treated fairly and the separate-package rationale is credible. The comparison remains mostly prose rather than explicit capability dimensions, and it omits a directly comparable Python option. |
| Scientific and numerical accuracy | 144/150 | The reported health-example values match the generated evidence, and the normal--normal model is now declared. The paper does not state enough assumptions to let a reader interpret the EVSI result without consulting external files. |
| Software design and research relevance | 96/100 | The architecture is tied to research use and its trade-off is clear. The promotion-ladder paragraph is still partly internal governance description. |
| Reproducibility, packaging, documentation, and tests | 82/100 | Local validation is strong, but the described worktree is uncommitted and unreleased; `v1.0.0` lacks the new EVSI and R fixes; no current official build is revision-bound; R and Julia require a separately supplied native library. |
| Research-impact statement | 61/80 | The example is concrete and honest, but it is synthetic and author-created. The interoperability artefact demonstrates engineering integration, not realised research impact. |
| Structure, metadata, and JOSS format | 58/60 | All required sections and metadata are present and the length is compliant. The date precedes this round, and the availability section contains future workflow text. |
| Clarity, accessibility, and sentence quality | 52/55 | The prose is restrained and mostly plain. “Normal--normal,” “numerical-parity,” “fixture-backed,” “backend-dependent,” and “CycloneDX” are not translated for non-specialists. |
| Citations, provenance, declarations, and AI disclosure | 20/25 | Field citations, funding, conflicts, authorship, archive, and disclosure are present. The exact software revision is missing; the AI statement says “submitted manuscript” before JOSS submission and cannot provide exact historical model versions. |
| **Total** | **918/1000** | **Major revision; fail-closed cap applies.** |

## Manuscript blocker

### B1. The cited release does not contain the software described

Lines 42--44 and 101--104 describe the new Rust analytical normal--normal EVSI
path. The current worktree contains that implementation, but tag `v1.0.0` does
not. The current R native-library correction also postdates `v1.0.0`. Lines
157--162 identify release 1.0.0 and promise a future revision freeze rather
than identifying the software actually reviewed.

Required resolution:

1. commit the complete scientific, binding, documentation, and manuscript
   revision;
2. run the official build and full required test matrix on that commit;
3. create a release containing the described implementation, or rewrite the
   paper strictly to the capabilities of `v1.0.0`;
4. cite the exact tag and immutable commit reviewed by JOSS;
5. generate the revision-bound evidence manifest and archive record; and
6. replace the future-tense sentence at lines 161--162 with factual identifiers.

Suggested final form:

> Version X.Y.Z, corresponding to commit `<full commit>`, is the version
> reviewed for this paper. The fixed-seed script and synthetic outputs are
> included in that release, and the release-evidence manifest records their
> checksums.

The exact identifiers must be inserted only after they exist.

## Submission-eligibility finding separate from the manuscript score

Issue #471 remains open and has no attributable non-author report. The paper is
appropriately explicit about that absence. However, current JOSS screening
requires demonstrated research use at least by the developers, with external
adoption or integration a stronger signal. The synthetic example establishes
functionality and a possible use, while the same-author interoperability
contract establishes transfer between repositories. Neither is presently
described as a completed research analysis.

Before submission, provide one of:

- an attributable publication, preprint, report, protocol, or ongoing research
  workflow that actually used `voiage`; or
- editor-verifiable evidence of a real developer research analysis, described
  accurately and without turning the JOSS paper into a results paper.

Independent installation evidence remains strongly advisable but should not be
manufactured or conflated with research use.

## Paragraph audit

| Lines | Assessment | Defect or required action |
| ---: | --- | --- |
| 30--37 | Pass with minor context gap | Accessible opening and correct high-level definitions. The semicolon list is dense but functional. |
| 39--45 | Pass in prose; release mismatch | The binding boundary is now precise. The normal--normal EVSI statement is not true of the cited `v1.0.0` release. |
| 49--55 | Pass | The transfer problem is concrete and explains why arrays alone are insufficient. |
| 57--63 | Minor revision | The audience is clear. The claim that the same structure represents demand and environmental outcomes is uncited and unsupported by a worked example here; narrow or cite it. |
| 67--76 | Pass with minor comparison gap | The comparator set is much improved and appropriately respectful. A directly comparable Python package or an explicit explanation for limiting the comparison to established health-economic tools would complete the field account. |
| 78--84 | Pass | The build-versus-contribute rationale now rests on a language-neutral requirement and maintenance consequences. |
| 88--95 | Pass | The architecture and trade-off are explained in terms of shared calculations and parity maintenance. |
| 97--104 | Minor revision | Scientifically responsible, but “promotion ladder,” “fixture-backed,” and “backend-dependent” are internal terms. Retain the scientific boundary in plainer language. |
| 106--113 | Pass with release qualification | Accurate assurance and asset wording. The test catalogue is long, and the current evidence manifest is not yet bound to an immutable reviewed revision. |
| 117--128 | Pass scientifically; impact remains limited | Values match the current CSVs. State the study's key prior and outcome-variance assumptions in one clause or direct readers to a compact table. |
| 130--135 | Pass as an honest boundary | Correctly identifies author-created interoperability and absent independent adoption. It does not establish realised research impact. |
| 139--148 | Minor correction | Disclosure is transparent. “Reviewed the submitted manuscript” is inaccurate for this unsubmitted JOSS paper; use “reviewed this manuscript.” |
| 152--153 | Pass | Funding and competing-interest declarations are direct and complete for the stated circumstances. |
| 157--162 | **Blocker** | Archive and synthetic-data facts are useful, but the exact reviewed revision and matching release are absent and future workflow language remains. |

## Complete sentence inventory

| # | Lines | Assessment | Action |
| ---: | ---: | --- | --- |
| 1 | 30--31 | Pass | Retain. |
| 2 | 31--34 | Pass | Retain; the practical-question framing is effective. |
| 3 | 34--37 | Pass | Retain. |
| 4 | 39--41 | Pass | Retain. |
| 5 | 41--42 | Pass | Retain. |
| 6 | 42--43 | Release-blocked | Retain only after the matching EVSI implementation is released and revision-bound. |
| 7 | 43--44 | Pass | Retain; “source packages” correctly avoids registry claims. |
| 8 | 44--45 | Pass | Retain. |
| 9 | 49--51 | Pass | Retain. |
| 10 | 51--52 | Pass | Retain. |
| 11 | 52--54 | Pass | Retain. |
| 12 | 54--55 | Pass | Retain. |
| 13 | 57--58 | Pass | Retain. |
| 14 | 58--60 | Pass | Retain. |
| 15 | 60--63 | Minor evidence defect | Narrow to demonstrated domains or add direct evidence for demand and environmental use. |
| 16 | 67--68 | Pass | Retain. |
| 17 | 69--70 | Pass | Retain; current CRAN metadata supports the claim. |
| 18 | 70--72 | Pass | Retain. |
| 19 | 72--74 | Pass | Retain; the direct SAVI source now supports the software statement. |
| 20 | 74--76 | Pass | Retain. |
| 21 | 78--80 | Pass | Retain. |
| 22 | 80--82 | Pass | Retain; this is a project-requirement argument, not speculation about other maintainers. |
| 23 | 82--84 | Pass | Retain; the limitation is appropriately prominent. |
| 24 | 88--89 | Pass | Retain. |
| 25 | 89--90 | Pass | Retain. |
| 26 | 90--91 | Pass | Retain. |
| 27 | 91--93 | Pass | Retain. |
| 28 | 93--95 | Pass | Retain. |
| 29 | 97--98 | Clarity defect | Replace internal ladder terminology with a brief explanation of what evidence is required before a method is presented as stable. |
| 30 | 98--99 | Clarity defect | Replace “surrogate-based” and “backend-dependent” with plain-language descriptions or omit the list. |
| 31 | 99--101 | Pass | Retain. |
| 32 | 101--103 | Pass but release-blocked | Scientifically specific; valid only after the matching implementation is released. |
| 33 | 103--104 | Pass | Retain; this accurately bounds compatibility estimators in the worktree. |
| 34 | 106--108 | Pass | Retain. |
| 35 | 108--110 | Minor style defect | Shorten the catalogue unless each test class is important to the submission argument. |
| 36 | 110--111 | Pass | Verified against the public `v1.0.0` assets. |
| 37 | 111--113 | Pass | Correctly separates attestations from the separate SBOM workflow. |
| 38 | 117--118 | Pass | Retain. |
| 39 | 118--120 | Pass | Retain. |
| 40 | 120--121 | Pass | Matches the fixed-seed summary CSV. |
| 41 | 121--122 | Pass | Matches the fixed-seed summary CSV. |
| 42 | 122--123 | Pass but release-blocked | Matches current generated evidence; release the implementation and evidence together. |
| 43 | 123--125 | Pass | Matches the base-case CSV. |
| 44 | 125--126 | Pass | Matches the delayed/60%-uptake CSV. |
| 45 | 126--128 | Pass | Restrained interpretation; retain. |
| 46 | 130--132 | Pass | Retain. |
| 47 | 132--133 | Pass | Essential non-adoption boundary; retain. |
| 48 | 133--135 | Pass | Accurate current external-evidence boundary. |
| 49 | 139 | Pass | Retain. |
| 50 | 139--142 | Pass | Tool names and assistance scope are stated. |
| 51 | 142--143 | Pass with unavoidable limitation | Transparent about missing exact historical model identifiers. |
| 52 | 143--146 | Claim defect | Replace “reviewed the submitted manuscript” with “reviewed this manuscript.” |
| 53 | 146--148 | Pass | Retain. |
| 54 | 152 | Pass | Retain. |
| 55 | 152--153 | Pass | Retain. |
| 56 | 157 | Pass but incomplete identity | Retain after replacing 1.0.0 with the release actually reviewed, if different. |
| 57 | 157--158 | Pass | Retain. |
| 58 | 158--161 | Pass | The SWHID is concrete; verify that the final reviewed release is included or archive a new snapshot. |
| 59 | 161--162 | **Blocker** | Replace the future promise with the exact tag, commit, manifest, and archive evidence after they exist. |

## Required changes before round 3

1. Resolve the version mismatch by releasing the exact software described or
   restricting the paper to the actual `v1.0.0` contents.
2. Replace the revision-freeze placeholder with immutable identifiers and a
   revision-bound evidence manifest.
3. Produce and visually inspect an official Open Journals build from that
   immutable revision.
4. Correct “reviewed the submitted manuscript.”
5. Either narrow the cross-domain sentence or support it with concrete evidence.
6. Translate or remove the internal maturity and approximation vocabulary.
7. Add enough context to interpret the normal--normal EVSI result without
   requiring source inspection.
8. Record genuine developer research use and, if obtained, attributable
   non-author validation without overstating either.

After items 1--7, the manuscript itself should be suitable for another
high-threshold panel review. Item 8 remains the principal JOSS
submission-eligibility risk and must be assessed separately from prose quality.
