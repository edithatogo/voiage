# Round 3 JOSS sentence-editor report

## Disposition

**Recommendation:** major revision; do not submit this revision to JOSS.

**Score:** **768/1000**.

**Fail-closed cap:** the score cannot exceed 799 while a manuscript sentence
misstates the scientific model, the method-maturity labels outrun the evidence,
or the paper identifies a future rather than an immutable reviewed release.

**Revision inspected:** committed base
`1dfcd3d42af591ca15fcc03ad958123cc153dbbf` on
`codex/joss-panel-review`, plus the uncommitted working tree present on
24 July 2026. The canonical JOSS source reviewed here is `paper.md`; this report
does not assess the separate arXiv manuscript.

This audit applies the current JOSS
[paper requirements](https://joss.readthedocs.io/en/latest/paper.html),
[review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html),
and [pre-review screening criteria](https://joss.readthedocs.io/en/latest/submitting.html),
together with the pinned Authentext academic patterns. Authentext is used as a
claim-hygiene and prose diagnostic, not as a detector of who or what wrote the
text.

## Score breakdown

| Dimension | Score | Maximum | Finding |
| --- | ---: | ---: | --- |
| JOSS sections, scope, and screening case | 190 | 220 | All required sections are present and the developer research use meets the stated minimum, but the single-author/no-independent-use record remains a screening risk. |
| Scientific accuracy and maturity consistency | 150 | 250 | The analytical EVSI example is coherent, but the generic built-in two-loop description is contradicted by the implementation and its independent numerical review. |
| Release, archive, and reproducibility precision | 70 | 150 | `v1.0.0` is public and archived, but it does not contain the revised EVSI contract; the corrected release is prospective, and the checked release-evidence record has no provenance object and says the SBOM was not attached. |
| Clarity and non-specialist accessibility | 185 | 200 | The paper is concise, restrained, and mostly accessible. A few architecture and maturity sentences remain policy-heavy. |
| Authentext claim and voice discipline | 136 | 140 | No promotional language, vague authority, synthetic drama, generic conclusion, or excessive hedging was found. |
| Citation and reference hygiene | 37 | 40 | All citation keys reconcile and the records are specific; two DOI resolver requests were access-blocked rather than substantively verified in this pass. |
| **Total** | **768** | **1000** | **Major revision; fail closed.** |

## Overall-to-sentence assessment

The paper now has a clear research-facing argument: VOI calculations depend on
the decision description and study assumptions, and `voiage` aims to preserve
those details while sharing selected calculations across language interfaces.
The related-work discussion is fair, the worked health example reports
assumptions and results, and the manuscript does not claim independent adoption
that has not occurred.

The principal defect is not style. It is a mismatch among manuscript, runtime,
contract, and release:

1. Lines 120--123 say that current value, prior-predictive study results, and
   posterior value all use one fitted multivariate-normal prior.
2. The current built-in implementation fits a Gaussian prior for updating but
   selects outer-loop truths from empirical PSA rows. For a nonlinear economic
   model, those operations target different prior models.
3. The v2 contract nevertheless labels this route `fixture-backed`.
4. The public software citation remains `v1.0.0`, which predates both the
   analytical Rust EVSI route and the revised generic contract.
5. The availability section promises a later release instead of naming the
   software actually under review.

The Gaussian conditioning equations themselves are sound, and the analytical
normal--normal result used in the worked example has a prespecified reference.
That does not validate the generic two-loop estimator described in the design
section. The manuscript should either omit the generic estimator from the JOSS
paper until it is coherent and released, or describe it only after the runtime,
tests, maturity metadata, immutable release, and archive agree.

## Complete sentence inventory

The inventory contains all **82 grammatical prose sentences** in `paper.md`,
including the six sentences in the figure caption. Front matter and headings
are assessed separately below. `Pass` means no sentence edit is required.
`Revise` means the sentence is defensible but should be made more exact or
accessible. `Block` means the sentence or the state it reports prevents
submission under the requested fail-closed standard.

| ID | Lines | Class | Sentence and finding |
| ---: | ---: | --- | --- |
| S01 | 32--34 | **Pass** | “Value of Information … before a choice is made.” Accurate, cited, and accessible. |
| S02 | 34--37 | **Pass** | “It asks four practical questions … worth the study's cost?” The questions are concrete rather than artificial signposting. |
| S03 | 37--40 | **Pass** | “These questions correspond … ENBS.” Every abbreviation is expanded at first use. |
| S04 | 42 | **Pass** | “`voiage` calculates these measures … results.” Clear high-level functionality. |
| S05 | 42--44 | **Pass** | “Its decision and result records can retain … inputs came from.” “Can retain” appropriately avoids claiming that every scalar entry point carries every field. |
| S06 | 44 | **Pass** | “Python is the broadest interface.” Direct and consistent with the documented surfaces. |
| S07 | 45--47 | **Block** | “Rust provides … EVSI for a two-arm study ….” This describes the unreleased working tree, not cited `v1.0.0`. Name the immutable reviewed release before retaining the claim. |
| S08 | 47--48 | **Pass** | “The R and Julia source packages … EVPI interface.” The limitation is explicit. |
| S09 | 48--49 | **Pass** | “Readers can therefore distinguish … language interface.” A useful consequence, stated without self-praise. |
| S10 | 53--55 | **Pass** | “Analysts use VOI … model-output formats.” Concrete problem and audience. |
| S11 | 55--56 | **Pass** | “Moving an analysis … numerical array.” Effective transition. |
| S12 | 56--57 | **Pass** | “A result depends on strategy names … numerical values.” Specific and intelligible. |
| S13 | 57--59 | **Pass** | “Population, time horizon … interpreted.” The list consists of decision-relevant factors, not filler. |
| S14 | 59--60 | **Pass** | “Losing this information … different decisions.” Proportionate consequence. |
| S15 | 62--63 | **Revise** | “`voiage` provides … each calculation validates ….” “Each calculation” is universal. Limit the claim to the structured/public calculations actually covered by tests, or cite the validation contract. |
| S16 | 63--65 | **Pass** | “The package is intended … wider evidence assessment.” Audience and tasks are clear. |
| S17 | 66--67 | **Pass** | “Decision descriptions accept … health outcomes.” Bounded cross-domain rationale. |
| S18 | 67--69 | **Pass** | “For example … health economics.” Hypothetical domains are marked as possibilities and separated from demonstrated use. |
| S19 | 73--74 | **Pass** | “VOI analysis is well established ….” Directly cited field positioning. |
| S20 | 75--76 | **Pass** | “The R package `voi` ….” Specific comparison with a software citation. |
| S21 | 76--78 | **Pass** | “`BCEA` combines … and `dampack` supports ….” Concrete capabilities and citations. |
| S22 | 78--80 | **Pass** | “The Sheffield Accelerated … provides ….” SAVI is expanded and its scope is bounded. |
| S23 | 80--82 | **Pass** | “A review of web-based VOI tools ….” Specific peer-reviewed support, not vague attribution. |
| S24 | 82--84 | **Pass** | “These tools remain appropriate ….” Fair treatment of alternatives. |
| S25 | 86--88 | **Pass** | “Python also has specialised alternatives … noisy signal.” Scope and immutable software citation are specific. |
| S26 | 88--90 | **Pass** | “The author's `trd-cea-toolkit` ….” Same-author relationship is disclosed. |
| S27 | 92--94 | **Revise** | “`voiage` instead keeps … when an analysis moves between Python, R, and Julia.” R and Julia currently expose only narrow EVPI paths, so “an analysis moves” is broader than demonstrated portability. Name the records and EVPI result that cross the boundary. |
| S28 | 94--96 | **Revise** | “The project therefore uses … one language-specific workflow ….” This is the build-versus-contribute justification, but it does not identify why contributing the contracts to `voi`, `BCEA`, or `dampack` was unsuitable. Add one research-facing reason rather than a code-ownership reason. |
| S29 | 96--98 | **Pass** | “The trade-off is less method-specific depth ….” Candid and proportionate. |
| S30 | 102--103 | **Pass** | “The design separates calculations … language-specific.” Clear organising choice. |
| S31 | 103--104 | **Pass** | “Rust implements common types ….” Bounded by “selected”. |
| S32 | 104--105 | **Pass** | “Python provides the broader workflow ….” Accessible explanation of the Python role. |
| S33 | 105--107 | **Pass** | “The R and Julia packages call ….” Exact current limitation. |
| S34 | 107--109 | **Revise** | “This avoids maintaining separate EVPI implementations ….” Replace the abstract “This” and “agreement checks” with the concrete trade-off: a shared calculation reduces duplication, while each binding still needs installation and numerical-parity tests. |
| S35 | 111--113 | **Block** | “A method is labelled stable only when ….” The rule is sensible, but the current v2 metadata calls the scientifically inconsistent generic route `fixture-backed`. The labels must satisfy the stated rule before this sentence is true of the project. |
| S36 | 113--114 | **Pass** | “The documentation also says … optional software.” Clear disclosure policy. |
| S37 | 114--116 | **Pass** | “A working implementation therefore does not imply ….” Necessary scientific restraint. |
| S38 | 116--120 | **Pass** | “The analytical EVSI calculation applies ….” The assumptions match the Rust analytical model and the worked example. |
| S39 | 120 | **Pass** | “These assumptions determine … uncertainty.” Short explanatory bridge. |
| S40 | 120--123 | **Block** | “The built-in two-loop model … under that same prior.” False for the current implementation: outer truths are selected from empirical PSA rows while posterior updating uses a fitted Gaussian prior. Choose and implement one prior interpretation before making this claim. |
| S41 | 123--124 | **Block** | “This developing estimator uses moment-matched … negative estimate.” Moment matching does not remove nested decision-maximisation bias, and reporting only a negative estimate does not supply Monte Carlo uncertainty or a convergence diagnostic. The sentence overstates scientific maturity. |
| S42 | 125--126 | **Pass** | “Analysts can supply … other study models.” The paired callback route exists; its scientific validity remains the analyst's declared model responsibility. |
| S43 | 126--128 | **Block** | “The analytical model is the stable … generic … not labelled stable ….” The analytical route is not in the cited release, and the generic route is simultaneously called `fixture-backed` in the v2 contract. Release and maturity terms must agree. |
| S44 | 130--132 | **Revise** | “The repository tests calculations … supported operating systems.” Broad but plausible; identify this as current repository assurance rather than evidence attached to the reviewed release. |
| S45 | 132--133 | **Pass** | “The v1.0.0 release contains ….” The checked release manifest records one sdist, three wheels, and `SHA256SUMS`. |
| S46 | 133--135 | **Block** | “GitHub records build-provenance attestations … inventory ….” The checked `v1.0.0` evidence has `provenance: null` and says its SBOM was not attached. The broader mixed-language SBOM workflow is also an unpublished working-tree change. State only evidence bound to the reviewed release. |
| S47 | 139--140 | **Pass** | “The repository contains one synthetic … same-author integration ….” Exact and candid scope. |
| S48 | 140--142 | **Pass** | “The health example compares … uncertain.” Clear decision problem. |
| S49 | 142--146 | **Pass** | “Across 10,000 … health gain … programme cost ….” Distributional assumptions and independence are explicit. |
| S50 | 146--147 | **Pass** | “In this synthetic example … value units ….” Prevents a false currency claim. |
| S51 | 147--149 | **Pass** | “At 50,000 … incremental net benefit ….” Defines the quantity in plain language. |
| S52 | 149 | **Pass** | “The proposed study informs health gain, not programme cost.” Essential sampling-model boundary. |
| S53 | 149--151 | **Pass** | “It has equal allocation … 50 to 1,200.” Study design is inspectable. |
| S54 | 153--154 | **Pass** | “The programme is preferred in 49.2% ….” Point estimate and bootstrap interval match the machine-readable summary. |
| S55 | 154--156 | **Pass** | “Estimated EVPI is 644 … EVPPI ….” “Estimated” is retained and intervals match the generated data. |
| S56 | 156--157 | **Pass** | “For a total sample of 200 … EVSI of 124 ….” The analytical result matches the prespecified Rust reference, subject to publication of the implementation. |
| S57 | 157--160 | **Pass** | “ENBS assumes 1,300 … cost of 100 ….” All assumptions needed for the reported sign changes are stated. |
| S58 | 160--161 | **Pass** | “With immediate full uptake ….” The evaluated points match the base-case data. |
| S59 | 161--162 | **Pass** | “With a two-year delay and 60% uptake ….” The evaluated points are explicit rather than presented as a continuous threshold. |
| S60 | 162--164 | **Pass** | “The synthetic example illustrates … not … real trial.” Correct restraint. |
| S61 | 166--170 | **Pass** | Caption opening: the three panel-level findings are faithful to the plotted values and remain descriptive. |
| S62 | 170 | **Pass** | “In panel A … 50,000 ….” Reference line is defined. |
| S63 | 171 | **Pass** | “In panel B … all uncertainty.” Dashed line is defined. |
| S64 | 172--173 | **Pass** | “In panel C … exceeds study cost.” Zero line is interpreted correctly. |
| S65 | 173 | **Pass** | “EVPI and EVPPI use 10,000 … EVSI is analytical.” Estimator distinction is explicit. |
| S66 | 174 | **Pass** | “All inputs are synthetic.” Unambiguous data boundary. |
| S67 | 176--179 | **Pass** | “The developer-led … `vop_poc_nz` … uses a versioned … bundle ….” The citation points to an immutable commit and named contract path. |
| S68 | 180--181 | **Pass** | “This is use in a research workflow … not … independent adoption.” Meets the JOSS minimum developer-use signal without overclaiming. |
| S69 | 181--182 | **Pass** | “The package has been developed publicly since July 2025.” The local history begins 3 July 2025; retain only if the public-hosting history confirms the same date at submission. |
| S70 | 182--183 | **Pass** | “Attributable non-author use has not yet been documented.” Candid limitation. It creates screening risk for a single-author project but is preferable to an unsupported adoption claim. |
| S71 | 187 | **Pass** | “Generative artificial intelligence … assisted ….” AI is expanded at first use and use is disclosed directly. |
| S72 | 187--190 | **Pass** | “OpenAI Codex … Google Jules … assisted ….” Tools and activities are specific; the historical model boundary is appropriately qualified in S73. |
| S73 | 190--191 | **Pass** | “Exact model identifiers were not retained ….” Transparent limitation. |
| S74 | 191--194 | **Block** | “The human author selected … checked … against … evidence.” JOSS requires confirmation that the human author reviewed, modified, and validated all AI-assisted outputs and made the primary design decisions. “Checked the reported” subset is narrower, and the repository validator therefore fails. Replace only with a statement the author can personally attest. |
| S75 | 194--196 | **Pass** | “The author accepts responsibility … no AI system is an author.” Clear accountability and authorship boundary. |
| S76 | 200 | **Pass** | “This work received no external funding.” Direct declaration. |
| S77 | 200--201 | **Pass** | “The author declares no competing interests.” Direct declaration. |
| S78 | 205 | **Pass** | “The Python package and release 1.0.0 are public.” True, specific, and cited. |
| S79 | 205--207 | **Pass** | “The fixed-seed health-example script … use synthetic data.” The script and outputs exist; add the actual seed only if space permits. |
| S80 | 207--210 | **Revise** | “The repository is preserved … [SWHID].” The SWHID is evidence for a particular snapshot, not the changing repository in general. Say that Software Heritage preserves the `v1.0.0` snapshot identified by this SWHID. |
| S81 | 210--211 | **Block** | “Release 1.0.0 predates the revised EVSI contract described here.” Factually candid, but it proves that the cited software is not the software described by the paper. Do not submit until one immutable reviewed release contains the claimed functionality. |
| S82 | 211--212 | **Block** | “The submitted paper will cite a release made ….” Prospective workflow language is not availability evidence. Replace with the exact reviewed version, tag, commit, archive identifier, and evidence-manifest location after publication and archival. |

### Inventory reconciliation

The 82 entries above map one-to-one to the grammatical sentences in the current
file after Markdown line wrapping is removed. Semicolon-separated independent
clauses remain part of their containing grammatical sentence. The caption's
alt text is included because it is public manuscript prose. No body sentence
was sampled or omitted.

## Front matter and heading audit

- **Pass:** title is descriptive, domain-facing, and non-promotional.
- **Pass:** author, three affiliations, ORCID, repository, and bibliography
  fields are present.
- **Revise at submission:** regenerate the date so that it represents the
  reviewed submission revision rather than an earlier editing date.
- **Pass:** all JOSS-required headings are present: Summary, Statement of need,
  State of the field, Software design, Research impact statement, AI usage
  disclosure, Acknowledgements, and References.
- **Pass:** “Software and data availability” is a useful additional section and
  does not contain API documentation.
- **Pass:** the repository validator confirms that the body remains within
  JOSS's 750--1750-word range; the shell `wc` total of 1,576 includes YAML
  metadata.

## Authentext findings

| Pattern | Result |
| --- | --- |
| A1, vague literature attribution | **Pass.** Claims about the field and tools name specific sources. |
| A2, formulaic literature review | **Pass.** Comparisons are concrete and restrained. |
| A3, excessive hedging | **Pass.** Qualifications mark real evidence boundaries. |
| A4, generic conclusion | **Pass.** No generic conclusion is present; the factual availability section closes the paper. |
| A5, promotional language | **Pass.** No “novel”, “groundbreaking”, “comprehensive”, “best”, or equivalent marketing claim appears. |
| A6, methodology filler | **Pass with minor revisions.** Most sentences carry substantive information; S34 and S44 can be made more direct. |
| A7, artificial signposting | **Pass.** No mechanical “firstly/secondly” structure or paper roadmap appears. |
| A8, vague quantitative claims | **Pass.** The worked example names draws, distributions, sample sizes, estimates, intervals, and ENBS assumptions. |
| A9, inaccurate or fabricated citation | **No detected key mismatch.** Citation existence and metadata still require normal human/source verification; DOI access checks alone are not content verification. |
| A10, citation padding | **Pass.** Citations are attached to claims that need them. |
| Reasoning/self-consistency | **Block.** S40, S41, S43, S46, S81, and S82 conflict with runtime, maturity, or release evidence. |

The prose shows no obvious Authentext surface-pattern cluster. The remaining
problems are evidence and contract problems, not an “AI voice” problem.

## Lightweight checks run

| Check | Result |
| --- | --- |
| `uv run python scripts/validate_joss.py` | **Failed:** “AI disclosure must record human review and validation.” |
| `uv run --extra ci --extra dev pytest tests/test_joss_readiness.py tests/test_evsi_scientific_contract.py --no-cov -q` | **Failed:** three JOSS-readiness tests failed; the scientific tests in this focused invocation did not fail. Passing tests do not cure the untested hybrid-prior defect. |
| `uv run bash scripts/vale_prose.sh paper.md` | **Passed:** 0 errors, 0 warnings, 0 suggestions. |
| Citation-key reconciliation | **Passed:** 14 cited keys and 14 bibliography keys; no missing or uncited record. |
| DOI/URL reachability spot check | Six DOI targets returned HTTP 200; the two SAGE DOI targets returned HTTP 403 to the automated request. The public `v1.0.0` release URL returned HTTP 200. |
| Public-development history | Local Git history begins 3 July 2025, more than six months before this review. Public-host chronology should still be checked at submission. |
| `v1.0.0` release evidence | Records one sdist, three wheels, and checksums; records `provenance: null`; records the SBOM as not attached to the GitHub release. |
| v2 contract gate | Explicitly says runtime and binding conformance are required and that the contract alone does not establish a published v2 release. |

## Prioritised changes

### P0 — submission blockers

1. **Make the generic built-in EVSI model coherent.** Choose one prior
   interpretation. For a fitted joint Gaussian contract, generate outer study
   data from the fitted prior predictive distribution and calculate current and
   posterior values under that same fitted prior. Alternatively, implement an
   empirical-prior posterior-weighting model. Add a non-Gaussian counterexample,
   convergence checks, Monte Carlo uncertainty, and EVSI/EVPI-bound tests.
2. **Correct the maturity metadata.** Until item 1 and its validation matrix
   pass, do not call the generic route `fixture-backed` or imply that it meets
   the project's stable-label rule.
3. **Publish one immutable reviewed release.** The release must contain the
   analytical Rust EVSI route, any generic estimator described in the paper,
   synchronized Python/R/Julia manifests, tests, package artifacts, checksums,
   provenance, and the accurately scoped SBOM.
4. **Archive that release and replace S81--S82.** Cite the exact version, tag,
   commit, release URL, Software Heritage identifier, and evidence manifest.
   Remove every future-tense publication promise.
5. **Repair the AI disclosure through author attestation.** State the JOSS
   required human review, modification, validation, and primary-design
   responsibility only if the author can personally confirm each point.
6. **Re-run the JOSS validator and focused/full test suites.** All repository
   checks must pass at the immutable revision; the current three JOSS-readiness
   failures are not editorially waivable.

### P1 — screening and editorial strength

7. Make the build-versus-contribute argument research-facing: identify the
   language-neutral provenance/result contract and explain why adding it to one
   specialist R workflow would not meet the cross-language research need.
8. Narrow the portability claim to what R and Julia actually share today.
9. Keep the candid same-author-use statement, but add attributable external
   use or engagement if it becomes available before submission. Do not convert
   automated-agent activity into community evidence.
10. Replace broad assurance catalogues with evidence bound to the reviewed
    release.

### P2 — sentence polish

11. Replace the abstract subject in S34 and narrow the universal quantifier in
    S15.
12. Update the front-matter date only when the immutable submission revision is
    selected.

## Acceptance condition for the next sentence-editor round

A later sentence review may score above 995 only when:

- all 82 sentences, or their replacements, are `Pass`;
- the coherent generic EVSI target and its uncertainty are tested rather than
  inferred from passing conjugate-moment tests;
- method maturity labels match the evidence;
- the manuscript, versioned contracts, runtime, language bindings, release
  artifacts, provenance, SBOM, and archive identify the same immutable
  revision;
- JOSS validation, prose lint, citation reconciliation, and the relevant test
  matrix pass; and
- no sentence depends on a future release, future archive, unsupported use, or
  unpublished scientific behaviour.
