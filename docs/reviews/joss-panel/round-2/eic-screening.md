# Simulated JOSS Associate Editor-in-Chief screening: round 2

Review date: 24 July 2026

Repository: `edithatogo/voiage`

Reviewed worktree branch: `codex/joss-panel-review`

Reviewed checked-out commit: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`

Worktree state: materially modified and uncommitted

Role simulated: JOSS Associate Editor-in-Chief

Status: internal AI-assisted review only. This is not a JOSS decision, does not
represent an Open Journals editor, and does not predict acceptance.

## Screening disposition

**Do not proceed to JOSS submission from this snapshot.**

Round 2 resolves most of the scientific and wording defects identified in
round 1. The current source now has a declared normal--normal expected value of
sample information (EVSI) calculation, a corrected two-loop posterior update,
explicit warnings for compatibility estimators without a study likelihood, and
a working installed R-to-Rust expected value of perfect information (EVPI)
path. The worked-example values agree with the executable evidence inspected.
The manuscript is also substantially clearer about the unequal Python, R, and
Julia surfaces and no longer presents author-created demonstrations as
independent adoption.

The submission nevertheless fails closed for three reasons:

1. **The demonstrated-research-impact pre-review gate is not met by the
   evidence presented.** The manuscript accurately calls the two examples
   demonstrations and states that attributable non-author research use has not
   been documented. The current JOSS guidance makes actual research use a hard
   pre-review gate, at minimum by the developers. A synthetic worked example
   and a same-author interoperability fixture establish research readiness, but
   neither is documented as use of the software in an actual research
   analysis or research workflow.
2. **The software described by the revised paper is not available as a frozen,
   reviewer-installable revision.** The public `v1.0.0` release points to
   `05cc373d78ae74143194e889ff1317de4dfea52e`. It predates the round-2
   normal--normal EVSI API, two-loop correction, R native-symbol repair,
   manuscript revision, and current evidence files. A clean installation of
   `voiage==1.0.0` confirmed that EVPI works but
   `normal_normal_two_arm_evsi` is absent. The checked-out fixes are uncommitted,
   have no immutable reviewed commit, and have not passed hosted checks at that
   revision.
3. **Citation and submission metadata remain unresolved.** The Rothery et al.
   record names `John F. Murray`; Crossref, PubMed, and the article identify
   `James F. Murray`. The final availability sentence also says that the exact
   reviewed revision “will be frozen before submission,” which is an unresolved
   workflow placeholder rather than submission-ready availability information.

The raw rubric score is **852/1000**. Material release/revision and citation
defects activate the rubric's fail-closed rule. The raw score is already below
the 950 cap, so the capped score remains **852/1000**. It does not meet the
repository's requested threshold of 996/1000.

## Standard applied

This review uses:

- the repository's
  [fail-closed rubric](../rubric.md);
- the [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html);
- the [JOSS paper format](https://joss.readthedocs.io/en/latest/paper.html);
- the [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html);
- the [JOSS editorial guide](https://joss.readthedocs.io/en/latest/editing.html);
- the round-1 [synthesis](../round-1/synthesis.md);
- the repository-local Authentext academic, claim-hygiene, and reasoning
  checklists; and
- the pinned SourceRight validation path in read-only, disposable-output mode.

The current JOSS requirements distinguish two related points that the
repository's readiness material partly conflates:

- Demonstrated research impact is a hard pre-review gate. JOSS asks for
  evidence that the software is actually being used for research, at minimum by
  its developers.
- Non-author issues, pull requests, discussions, adoption, or validation are a
  strong positive signal. They are not stated as a universal hard gate, although
  the reviewer guidance describes a single-author project with no community
  engagement, external use, or collaborative input as not acceptable.

Issue #471 is therefore useful and material, but its current wording is stricter
than the first gate alone. A genuine developer research use could satisfy the
minimum impact gate; attributable non-author evidence would address the
separate single-author engagement risk much more convincingly.

## Evidence inspected

### Manuscript and bibliographic evidence

- `paper.md`
- `paper.bib`
- `docs/reviews/joss-panel/rubric.md`
- `docs/reviews/joss-panel/round-1/synthesis.md`
- all eight round-1 role reports
- `CITATION.cff`
- `codemeta.json`
- `docs/release/joss-submission-readiness.md`
- `docs/release/joss-independent-validation.md`

### Scientific and executable evidence

- `rust/crates/voiage-numerics/src/evsi_normal_normal.rs`
- `rust/crates/voiage-numerics/tests/evsi_normal_normal.rs`
- `voiage/methods/sample_information.py`
- `tests/test_evsi_scientific_contract.py`
- `tests/test_sample_information.py`
- `scripts/generate_paper_health_example.py`
- the three synthetic-health-example CSV outputs
- `tests/test_paper_health_example.py`
- the stable API and maturity contracts under `specs/v1/` and
  `specs/core-api/`

### Binding, packaging, and release evidence

- the Rust workspace and C ABI
- the Python runtime adapter
- `r-package/voiageR`, including its installed native test
- `bindings/julia`, including its shared numerical-reference tests
- `.github/workflows/bindings-ci.yml`
- `.github/workflows/joss-paper.yml`
- `.github/workflows/sbom.yml`
- `docs/release/v1.0.0-release-evidence.json`
- `docs/release/v1.0.0-sbom.cdx.json`
- `docs/release/release-evidence.md`
- the live GitHub `v1.0.0` release asset inventory
- GitHub's commit-verification record for the release commit
- the Software Heritage snapshot identifier
- live issue #471 and its single maintainer-authored comment
- local commit distribution and contributor history

## Verification results

### Passed on the current worktree

- `uv run --extra ci python scripts/validate_joss.py`
- `uv run --extra ci pytest tests/test_joss_readiness.py --no-cov -q`
  reported 7 passing tests.
- `uv run --extra dev tox -e joss` passed.
- `uv run --extra dev tox -e vale` reported no errors, warnings, or
  suggestions across its configured prose set.
- `uv run --extra ci --extra dev pytest
  tests/test_evsi_scientific_contract.py tests/test_sample_information.py
  --no-cov -q` reported 46 passing tests.
- `cargo test --manifest-path rust/Cargo.toml -p voiage-numerics --test
  evsi_normal_normal` reported 3 passing tests.
- `uv run --extra ci --extra dev pytest tests/test_paper_health_example.py
  --no-cov -q` reported 5 passing tests.
- The release-evidence validator accepted
  `docs/release/v1.0.0-release-evidence.json` with the documented
  missing-local-assets allowance.
- A clean local installation of the current R source package called the
  separately built Rust library and returned EVPI `0.5`.
- The current Julia source package passed 12 tests, including all shared EVPI
  reference cases.
- `git diff --check` found no patch errors. It did report future
  CRLF-to-LF conversion warnings for the three generated CSV files.
- The JOSS body contains 1,000 words, within the required 750--1,750 range.

### SourceRight and reference checks

Pandoc converted all 11 BibTeX records into disposable CSL JSON. SourceRight
reported:

- five non-canonical mixed-case DOI values; and
- six empty CSL types produced from `@misc` records.

After lower-casing DOI values and assigning `software` to the empty converted
types in the disposable copy, SourceRight returned no structural diagnostics.
This repeats the known Pandoc-to-CSL normalisation limitation from round 1; no
writeback was applied.

The repository validator found no missing citation keys and no uncited
bibliography records. Crossref metadata agreed with the principal article and
package records except for one material author-name defect:

- `paper.bib` records `Murray, John F.` for Rothery et al.;
- Crossref, PubMed, and the article record `James F. Murray`.

The package-year fields for the CRAN software citations appear to describe
specific package releases rather than the year when the persistent CRAN DOI
record was first deposited. That distinction should be checked when the
bibliography is corrected, but it is not treated as a proven error here.

### Failed or unavailable for the reviewed snapshot

- The current worktree is not an immutable revision. Its scientific, binding,
  manuscript, workflow, and release-evidence changes are uncommitted.
- No current JOSS PDF exists in the worktree, and no hosted Open Journals build
  can correspond to the dirty snapshot.
- The public `v1.0.0` wheel does not contain
  `normal_normal_two_arm_evsi`.
- The `v1.0.0` R source has the round-1 broken
  `PACKAGE = "voiageR"` symbol lookup rather than the current repair.
- The checked-in CycloneDX file is revision-bound to the older `v1.0.0` tag,
  not to the present worktree.
- The current R three-operating-system workflow has not run against an exact
  committed round-2 revision.
- Issue #471 remains open with no non-author report.

These are evidence failures or unavailable evidence. They are not inferred to
be code failures where the corresponding current local test passed.

## Round-1 disposition

| Round-1 convergent finding | Round-2 disposition |
| --- | --- |
| Public EVSI claim exceeded scientific evidence | **Resolved in current source, unreleased.** The declared analytical model, corrected two-loop update, warnings, and reference tests pass. |
| Installed R package did not execute Rust EVPI | **Resolved in current source, unreleased.** A clean installed-package test returned `0.5`; the public tagged source still contains the defect. |
| Demonstrations were described as research uses | **Resolved as a wording defect.** The paper now calls them demonstrations and disclaims independent adoption. The underlying JOSS research-impact gate remains unmet. |
| Cross-language claims exceeded the EVPI-only shared surface | **Mostly resolved.** The paper states that Python is broader and that R and Julia share EVPI. One phrase still understates the R reticulate compatibility wrappers and should identify the direct native surface precisely. |
| The release was said to contain an SBOM | **Resolved.** The paper now distinguishes release assets, attestations, and the separate SBOM workflow. |
| Field comparison omitted `dampack`, lacked a direct SAVI record, and misattributed `voi` | **Partly resolved.** All three additions/corrections are present. The comparison still omits a directly relevant Python implementation and software-review literature, and the Rothery record has a different author-name error. |
| Maturity terminology was outside the governed taxonomy | **Resolved.** The four-level promotion ladder and separate result-character metadata agree with current contracts. |
| Worked example omitted results | **Resolved.** The principal decision, EVPI, EVPPI, EVSI, and ENBS results are reported and verified. |
| Engineering terminology displaced the applied problem | **Mostly resolved.** The Summary and Statement of need are substantially more accessible. The promotion and assurance paragraphs remain dense for non-developers. |
| Availability material was misplaced and lacked an exact revision | **Partly resolved.** It has its own section and an SWHID, but the exact revision is still future-tense and absent. |

## Rubric scores

| Dimension | Maximum | Score | Deductions |
| --- | ---: | ---: | --- |
| Scope, significance, and research use | 180 | **135** | −30 because the current evidence does not establish actual developer or external research use and therefore fails the current hard impact gate; −10 because non-health significance is supported by generic data structures rather than a completed non-health research application; −5 for the unresolved single-author engagement risk |
| Statement of need and audience | 120 | **112** | −4 because the practical consequences of losing metadata are plausible but uncited; −4 because the intended cross-domain audience is broader than the demonstrated application |
| State of the field and build-versus-contribute case | 130 | **112** | −7 for omitting a directly relevant Python VOI implementation and the software-review literature identified in round 1; −6 because “required language-neutral interface” states an internal design requirement rather than evidenced user demand; −5 because the comparison is almost entirely health-economic while the package claims broader applicability |
| Scientific and numerical accuracy | 150 | **145** | −3 because the stable/provisional relationship between the generic two-loop path and analytical helper is difficult to infer from the paper; −2 because the rounded example results omit uncertainty intervals that are available in the generated evidence |
| Software design and research relevance | 100 | **95** | −3 because the research consequence of the promotion ladder is explained abstractly; −2 because the R source package also contains Python-backed compatibility wrappers, making “a narrower EVPI interface” imprecise |
| Reproducibility, packaging, documentation, and tests | 100 | **78** | −10 because the described fixes are uncommitted and absent from the public release; −5 because no exact revision or current official PDF exists; −4 because R and Julia require a separately built native library rather than ordinary registry installation; −3 because the present SBOM and release evidence describe the older tag rather than the reviewed worktree |
| Research-impact statement | 80 | **46** | −25 because the materials are demonstrations rather than documented research use; −5 because the same-author integration is not shown to have affected a research result or workflow; −4 because no independent or collaborative input is evidenced |
| Structure, metadata, and JOSS format | 60 | **58** | −2 for the unresolved future-tense reviewed-revision placeholder; required sections and word count otherwise pass |
| Clarity, accessibility, and sentence quality | 55 | **52** | −2 for the dense promotion/assurance paragraphs; −1 for repeated abstract nouns around contracts, provenance, and boundaries |
| Citations, provenance, declarations, and AI disclosure | 25 | **19** | −3 for the incorrect Rothery author name; −1 for the mutable, unversioned `vop_poc_nz` citation; −1 because exact AI model versions are not provided, although the limitation is disclosed; −1 for unsupported practical and cross-domain assertions |
| **Raw total** | **1,000** | **852** | |

### Fail-closed result

The following rubric conditions apply:

- a material release/revision claim is unresolved because the software
  described by the paper is not frozen or available in the cited release;
- citation metadata is incorrect; and
- the final availability sentence is a submission-workflow placeholder.

These conditions cap the review at 950. The raw score is 852, so the final score
is **852/1000**.

## Gate classification

### Manuscript gates

| Gate | State | Evidence and required resolution |
| --- | --- | --- |
| Correct Rothery bibliography metadata | **Not ready** | Replace `John F. Murray` with `James F. Murray`, rerun citation reconciliation, and regenerate the PDF. |
| Exact reviewed revision in availability section | **Not ready** | Replace the future-tense sentence with an immutable commit, reviewed release, and revision-bound evidence record. |
| Research-impact section meets current JOSS gate | **Not ready** | The prose is honest, but it documents demonstrations rather than actual research use. Add attributable evidence of a genuine developer or external research workflow without turning the paper into a results paper. |
| AI disclosure wording | **Minor revision** | Replace “reviewed the submitted manuscript” with “reviewed this manuscript” unless an exact submission is identified. Retain the candid model-version limitation and specify any recoverable tool versions. |
| State-of-field completeness | **Revision advised** | Add the directly relevant Python implementation and a software-review source, or define and defend why they are outside the comparator set. |
| Sentence-level audit | **Not clear** | Findings remain at sentences 7, 15, 22, 29--37, 46, 50--52, 57, and 59 below. |

### Repository and release gates

| Gate | State | Evidence and required resolution |
| --- | --- | --- |
| Immutable round-2 revision | **Not ready** | Commit the current changes and identify the exact SHA. |
| Reviewer-installable software matches paper | **Not ready** | Publish a patch/minor release or provide an equally clear immutable installation path containing the scientific and R fixes. The existing `v1.0.0` package is not the reviewed software. |
| Hosted checks at reviewed SHA | **Not ready** | Run the normal fail-closed matrix, including current JOSS rendering and the three-OS R native smoke, at the exact submitted commit. |
| Current JOSS PDF | **Not ready** | Generate and visually inspect the Open Journals PDF from the exact submitted revision. |
| Revision-bound SBOM and release manifest | **Not ready** | Generate evidence for the reviewed release rather than relying only on a post-hoc record for the superseded `v1.0.0` source. |
| Python primary package | **Partly ready** | Public `v1.0.0` installs and EVPI returns the expected value, but the new stable scientific contract is absent. |
| Current R source binding | **Locally ready; hosted evidence pending** | Clean installation and native EVPI passed on macOS. The public tagged source is stale, and the current OS matrix has not run at a committed SHA. |
| Current Julia source binding | **Locally ready; distribution pending** | All local tests passed. Standalone installation still requires a separately built library and is not evidenced through Julia General/JLL packaging. |

### External or human evidence gates

| Gate | State | Interpretation |
| --- | --- | --- |
| Genuine research use | **Not evidenced** | This is a hard JOSS pre-review gate. It can be satisfied by documented developer research use; it need not be non-author use, but it must be real and attributable. |
| Non-author installation, use, or engagement under issue #471 | **Pending** | Strongly advisable for this single-author project and relevant to the reviewer criterion on collaborative effort. It is a strong positive signal, not a separately stated universal hard gate in the submission guide. |
| Authorship, affiliations, funding, and conflicts | **Author-confirmed** | The repository records the author's confirmation. A simulated reviewer cannot independently certify personal declarations. |
| arXiv permanent identifier | **Pending; not a JOSS gate** | JOSS permits preprints before, during, or after review. The author's preferred sequence may still make this an author-selected gate. |
| JOSS submission, editorial screening, review, and acceptance | **Not begun / external** | Repository preparation cannot satisfy these outcomes. |
| Acceptance-stage archival DOI | **Later external action** | JOSS requests a tagged reviewed release and Zenodo or Figshare DOI after successful review, not as a pre-review requirement. |

## Substantive sentence audit

Every substantive sentence in `paper.md` is inventoried below. “Verified”
means that the inspected source or executable evidence supports the sentence at
the stated level. “Qualified” means that the core statement is supportable but
needs narrower wording or more evidence. “Defect” denotes a required
correction. Line references describe the reviewed snapshot.

| No. | Lines | Status | Audit finding |
| ---: | --- | --- | --- |
| 1 | 30--31 | Verified | The definition is consistent with Rothery et al. and is understandable to a non-specialist. |
| 2 | 31--34 | Verified | The four practical questions accurately distinguish decision uncertainty, parameter importance, proposed-study information, and study cost. |
| 3 | 34--37 | Verified | The EVPI, EVPPI, EVSI, and expected net benefit of sampling expansions are correct. |
| 4 | 39--41 | Qualified | The package retains labels, units, draws, study/population assumptions, warnings, and provenance across relevant objects and outputs, but not every method carries every listed field. “While keeping” can be read as universal. |
| 5 | 41--42 | Verified | Python is the broadest and primary installable interface. |
| 6 | 42--43 | Verified for current source | Rust supplies common contracts and selected calculations, including EVPI and the newly added analytical normal--normal EVSI calculation. This is not true of the public `v1.0.0` release. |
| 7 | 43--44 | Qualified | Julia directly exposes EVPI. R directly exposes EVPI but also retains Python-backed EVPPI/EVSI compatibility wrappers. “A narrower EVPI interface” should refer specifically to the direct native surface. |
| 8 | 44--45 | Verified | The revised text makes the unequal shared and language-specific scope substantially clearer. |
| 9 | 49--51 | Verified | The listed fragmentation across packages, languages, web tools, and model outputs exists in the field and in the compared tools. |
| 10 | 51--52 | Verified | Moving a decision analysis requires more than transferring only a numerical array. |
| 11 | 52--54 | Verified | The named metadata can change the interpretation or scaling of a result. |
| 12 | 54--55 | Qualified | Loss of those fields can make results describe different decisions, but the manuscript provides no direct source or concrete example for this consequence. |
| 13 | 57--58 | Verified | Decision/result objects and validation paths are inspectable and reject multiple malformed or inconsistent inputs. |
| 14 | 58--60 | Verified | The intended audience is specific and appropriate for JOSS research software. |
| 15 | 60--63 | Qualified | The data structures can represent the listed quantities, but only the health application is demonstrated as an analysis. The sentence should avoid implying validated application across all named domains. |
| 16 | 67--68 | Verified | VOI is established in decision analysis and health economics, and the cited works support that statement. |
| 17 | 69--70 | Verified | The `voi` package supports the listed measure families. The authorship correction from round 1 is present. |
| 18 | 70--72 | Verified | The BCEA and `dampack` summaries are proportionate and supported by their cited software records. |
| 19 | 72--74 | Verified | The direct SAVI record and Strong et al. support the web interface and regression-based EVPPI description. |
| 20 | 74--76 | Verified | The acknowledgement that established alternatives remain appropriate is restrained and fair. |
| 21 | 78--80 | Verified as project intent | Contract/provenance preservation across language boundaries is a real design objective with repository evidence. |
| 22 | 80--82 | Qualified | A separate implementation would create parity maintenance, but “the required language-neutral interface” is an internally asserted requirement. The paper does not show external user demand or why contribution to an existing ecosystem was impracticable. |
| 23 | 82--84 | Verified | The paper accurately acknowledges less method-specific depth and substantially narrower R/Julia surfaces. |
| 24 | 88--89 | Verified | The current architecture separates shared calculations from language-specific data handling, modelling, plotting, and reporting. |
| 25 | 89--90 | Verified | Rust implements common types, validation rules, and selected calculations. |
| 26 | 90--91 | Verified | Python provides the broader labelled-data and user-model workflow. |
| 27 | 91--93 | Verified | Both source packages call the Rust EVPI calculation and do not reproduce the full Python interface. |
| 28 | 93--95 | Verified | Avoiding independent EVPI implementations trades duplicated numerical maintenance for packaging and parity-test costs. |
| 29 | 97--98 | Verified | The four maturity states match `voiage/governance.py` and the revised specifications. |
| 30 | 98--99 | Verified | Separate calculation-character metadata now distinguishes exact, approximate, surrogate-based, and backend-dependent results. |
| 31 | 99--101 | Verified | Working code alone does not establish suitability for every research question. This is an appropriate scientific boundary. |
| 32 | 101--103 | Verified for current source | The analytical helper and default normal-arm two-loop path now declare the listed model elements. The analytical helper remains provisional in the stable-symbol contract, which the paper does not explain. |
| 33 | 103--104 | Verified | Regression, efficient, and moment-based compatibility paths emit explicit non-stable warnings because they lack a declared likelihood/posterior update. |
| 34 | 106--108 | Verified | The inspected tests and workflows cover the listed assurance categories. |
| 35 | 108--110 | Verified with scope qualification | The repository contains all named test types, but not every test family necessarily covers every claimed method or operating system. |
| 36 | 110--112 | Verified | The live release has one source distribution, three platform wheels, and `SHA256SUMS`, with matching recorded digests. |
| 37 | 112--113 | Verified | GitHub provenance covers the four package artifacts; the separate workflow generates a CycloneDX SBOM. The paper no longer claims that the SBOM is attached to the release. |
| 38 | 117--118 | Verified | The repository contains the two accurately classified demonstrations. |
| 39 | 118--120 | Verified | The health example compares programme and current practice under uncertain effect and cost. |
| 40 | 120--121 | Verified | The fixed-seed output records a 49.24% probability that the programme is preferred at 50,000 value units per health unit. |
| 41 | 121--122 | Verified | The generated summary records EVPI 644.15, effect EVPPI 589.67, and cost EVPPI 249.59, which support the rounded values. |
| 42 | 122--123 | Verified for current source | The analytical normal--normal calculation returns approximately 124.179 for 200 participants. It is absent from public `v1.0.0`. |
| 43 | 124--125 | Verified | The immediate/full-uptake sensitivity changes sign between 100 and 200 participants. |
| 44 | 125--126 | Verified | The delayed/60%-uptake sensitivity changes sign between 800 and 1,200 participants. |
| 45 | 126--128 | Verified as an illustration | The reported scenarios demonstrate that the named inputs can alter the research decision in this synthetic example. They do not establish effects in empirical applications. |
| 46 | 130--132 | Qualified | The versioned integration bundle transfers the listed artefacts, but the bibliography points to the mutable repository root rather than an immutable release, commit, or contract bundle. |
| 47 | 132--133 | Verified | Both demonstrations are author-created and the manuscript correctly disclaims independent adoption. |
| 48 | 133--135 | Verified | Git history begins in July 2025, and live issue #471 contains no attributable non-author result. |
| 49 | 139 | Verified | Generative AI tools assisted with the work. |
| 50 | 139--142 | Qualified | The disclosure identifies tool families and work categories, but not exact model versions. It candidly explains this in the next sentence. |
| 51 | 142--143 | Verified as a limitation | The repository does not retain exact historical model identifiers for every session. This is transparent, though it falls short of an ideal version-complete disclosure. |
| 52 | 143--146 | Qualified | The author has confirmed responsibility and validation, but “reviewed the submitted manuscript” is ambiguous because the JOSS manuscript has not been submitted. Use “this manuscript.” |
| 53 | 146--148 | Verified as author declaration | The responsibility and non-authorship statement is complete and consistent with JOSS policy. |
| 54 | 152 | Verified as author declaration | No external funding is declared. |
| 55 | 152--153 | Verified as author declaration | No competing interests are declared. |
| 56 | 157 | Verified | Python package version 1.0.0 and its GitHub release are public. |
| 57 | 157--158 | Verified | The script uses a fixed seed and the committed machine-readable outputs are synthetic. The sentence would be more useful with the script path or immutable revision. |
| 58 | 158--161 | Verified | The cited Software Heritage snapshot exists and is recorded in the release-evidence manifest. |
| 59 | 161--162 | Defect | The exact reviewed revision is not frozen. Future-tense workflow text is not submission-ready availability evidence and accurately exposes a current blocker. |

## Authentext claim and prose audit

The manuscript is restrained overall. It avoids promotional adjectives,
superiority claims, dramatic framing, generic conclusions, and unsupported
claims of adoption. The strongest remaining prose issues are not cosmetic:

- “required language-neutral interface” treats the project's design premise as
  an externally established need;
- “governed promotion ladder,” “fixture-backed,” “backend-dependent,”
  “build-provenance attestations,” and “CycloneDX” form a dense cluster for a
  non-specialist reader;
- the assurance paragraph is a catalogue of test types rather than an
  explanation of what a researcher can trust;
- the prose uses several compound technical nouns in succession, especially
  around language boundaries, contracts, provenance, and parity; and
- the availability section ends with diff-anchored workflow language rather
  than describing a finished reviewed artefact.

No high-density promotional or formulaic AI-writing pattern was found. The
paper's remaining problems concern evidence, precision, and abstraction level,
not tone inflation.

## Required actions before round 3

1. Document one genuine research use. It may be a developer-led analysis or
   integration, but it must be an actual research workflow rather than a
   synthetic demonstration or packaging fixture. Keep the scientific results
   outside the JOSS paper except for a concise description of how the software
   was used.
2. Obtain the issue #471 non-author report if feasible. Treat it as evidence,
   not an endorsement and not a substitute for the hard developer-use gate.
3. Commit the complete round-2 changes and run all hosted checks at the exact
   SHA.
4. Publish a reviewed patch or minor release containing the normal--normal EVSI
   work, corrected two-loop model, R symbol fix, and associated documentation
   and tests. Do not cite `v1.0.0` as the reviewed software after describing
   features absent from that release.
5. Generate revision-bound release evidence and an SBOM for that release.
6. Build and visually inspect the official JOSS PDF from the immutable
   revision.
7. Correct `James F. Murray`, rerun SourceRight reconciliation, and use an
   immutable `vop_poc_nz` citation target.
8. Replace the final availability placeholder with the exact commit, release,
   archive, seed, script, and evidence-manifest locations.
9. Clarify that the R direct native surface is EVPI while EVPPI/EVSI remain
   Python-backed compatibility wrappers, or simplify the paper to discuss only
   direct shared calculations.
10. Add the directly relevant Python comparator and software-review source, or
    record a defensible reason for their exclusion.
11. Replace “reviewed the submitted manuscript” with “reviewed this
    manuscript” and add any recoverable AI tool/model versions.
12. Repeat the full panel against the immutable release candidate. Do not award
    credit for tests or builds that ran only on this dirty snapshot.

## Round-2 conclusion

The current source is much closer to a scientifically defensible JOSS
submission than the round-1 snapshot. The principal EVSI defect and the R
native-runtime defect have credible local fixes, and the manuscript now
describes its demonstrations, binding scope, and SBOM boundary more honestly.

The package is not ready for JOSS screening from the inspected snapshot. The
remaining obstacle is not another prose pass. It is the absence of documented
research use and of a frozen, released, hosted-verified software revision that
contains the corrections the paper now describes. Those conditions should be
resolved before another editorial simulation is run.
