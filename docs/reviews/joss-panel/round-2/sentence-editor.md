# Round 2 Authentext sentence review

## Disposition

**Recommendation:** major revision before JOSS submission.
**Independent score:** **884/1000**.
**Revision inspected:** `HEAD=1dfcd3d42af591ca15fcc03ad958123cc153dbbf`
plus the uncommitted Round 2 working-tree changes present on 24 July 2026.

This review applies the pinned repository Authentext academic module, its core
patterns, its reasoning-failure checks, and the panel rubric. Authentext is used
as an editorial diagnostic rather than as evidence that prose was or was not
written with artificial intelligence. The manuscript has low surface-pattern
density: it contains no promotional superlatives, sycophantic language,
chatbot preamble, fabricated rhetorical drama, excessive hedging, vague
literature attributions, or generic conclusion. Its strongest features are its
specific numerical results and its candid distinction between developer use
and independent adoption.

The remaining weaknesses are concentrated rather than pervasive. Several
sentences use dense catalogues or abstract demonstrative subjects such as
"This division", "This choice", "This avoids", and "These results". More
importantly, four sentence-level defects affect meaning or verifiability:

1. the paper describes a new analytical EVSI implementation while its
   availability section identifies the older `v1.0.0` release;
2. the ENBS results refer to "stated population and cost assumptions", although
   those assumptions are absent from the manuscript;
3. the availability section promises that the reviewed revision "will be
   frozen" instead of identifying an immutable revision already reviewed; and
4. the `rothery2020voi` bibliography record gives two authors' names
   incorrectly.

These defects trigger the rubric's fail-closed cap. The score is therefore not
an acceptance prediction, and it cannot exceed 950 until the blockers are
removed.

## Overall-to-sentence diagnosis

### Whole-paper argument

The paper's central argument is defensible:

> VOI results depend on the decision description and study assumptions as well
> as the numerical formula. `voiage` provides structured workflows for carrying
> those details between tools, while selected Rust calculations give Python, R,
> and Julia a shared numerical implementation.

The manuscript presents that argument in the right order: concept, need,
related work, design, worked example, impact boundary, declarations, and
availability. No additional conclusion is needed. The availability paragraph
can provide a factual close once it identifies the exact reviewed release.

The main abstraction-level mismatch occurs between the accessible first half
and the governance language at lines 102--119. "Stable", "independent numerical
reference", "optional computing backend", "compatibility estimators", and the
software-assurance catalogue arrive in quick succession. Each concept is
relevant, but the paragraph reads like repository policy rather than an
explanation of how a researcher can judge a result. This is Authentext reasoning
pattern R4.

### Paragraph and cadence findings

- The opening four-question sequence is justified by the four VOI measures and
  is not an artificial rule-of-four construction.
- The Statement of need is the clearest section. Its transfer problem is
  concrete, but the package response at lines 57--58 is too universal.
- The State of the field is fair to existing software. It still needs a direct
  review of web-based VOI tools and immutable citations for software records
  that currently point to moving branches.
- The design section repeats "selected calculation" and relies on abstract
  subjects. It can be shortened without losing architectural meaning.
- The worked example contains useful exact values. It omits assumptions needed
  to interpret ENBS and presents simulation estimates as if they were exact
  population quantities.
- The disclosure and declarations are complete, but one disclosure sentence
  compresses too many activities into a list.
- The final sentence is workflow language. A submitted paper should state what
  was reviewed, not what will be done later.

### Authentext pattern inventory

| Pattern | Finding |
| --- | --- |
| A1/A9, vague or inaccurate citation | No vague attribution. One critical bibliography metadata error remains in `rothery2020voi`; several software citations are mutable. |
| A2, formulaic literature review | Not present. The comparisons are specific and appropriately restrained. |
| A3/A5, over-hedging or promotional prose | Not present. |
| A6, methodological filler | Minor at lines 102--110 and 112--119, where governance details can be expressed more directly. |
| A8, vague quantitative claims | Exact point estimates are given, but their estimator status and simulation uncertainty are not consistently named. |
| Core 7/10/22/26, stock vocabulary, list cadence, filler, over-structuring | Low-to-moderate clustering in the architecture, assurance, worked-example interpretation, and AI-disclosure paragraphs. |
| Core 13 | The three `normal--normal` spellings use the double-hyphen alias. Replace them with plain descriptions such as "normal prior and normal likelihood". |
| Core 27, technical literals | Preserve package names, identifiers, version strings, DOI values, the ORCID, URLs, commit identifiers, and the SWHID exactly. |
| Core 38, diff-anchored writing | The future-tense availability sentence is process-anchored rather than state-based. |
| R4, abstraction mismatch | Present when the paper moves from applied consequences to internal maturity and assurance terminology. |
| R6/R7/R8, quantitative, self-consistency, and verification failures | Present in the unstated ENBS assumptions and release/revision mismatch. |

## Complete substantive-sentence inventory

The inventory contains all 68 prose sentences in `paper.md`, including both
sentences in the figure caption. YAML metadata and headings are assessed after
the table because they are not prose sentences. "Pass" means no substantive
sentence edit is required. "Revise" gives replacement text. "Blocker" is a
fail-closed defect.

| ID | Lines | Classification | Sentence-level action |
| ---: | ---: | --- | --- |
| S01 | 30--31 | **Revise: conceptual precision.** "Benefit of reducing uncertainty" can imply that uncertainty reduction itself, rather than the changed decision, creates value. | Replace with: "Value of information (VOI) analysis estimates the expected improvement in a decision outcome that could follow from resolving uncertainty before a choice is made [@rothery2020voi]." |
| S02 | 31--34 | **Revise: logical precision.** EVSI measures expected decision improvement under a sampling model, not a generic amount of uncertainty reduction. | Replace with: "It asks four practical questions: could uncertainty change the preferred choice; which uncertain quantities matter most; how much would a proposed study improve the decision; and would that improvement be worth the study's cost?" |
| S03 | 34--37 | **Pass.** The four questions map clearly to the four expanded abbreviations. | Retain. |
| S04 | 39--41 | **Revise: overbroad agent and noun catalogue.** Scalar entry points do not retain every listed field. | Replace with: "`voiage` calculates these measures from probabilistic decision models. Its structured workflows can retain strategy labels, units, uncertainty draws, study and population assumptions, warnings, and provenance with the resulting analysis." |
| S05 | 41--42 | **Pass.** The primary interface is stated directly. | Retain. |
| S06 | 42--43 | **Blocker: release ambiguity; implementation phrase.** The current tree contains the declared EVSI model, but `v1.0.0` does not. `normal--normal` is also an unnecessary technical compound for this audience. | After publishing and identifying the reviewed release, use: "Rust defines shared data rules and performs selected calculations, including EVPI and EVSI for a two-arm study with a normal prior and normal likelihood." |
| S07 | 43--44 | **Pass.** The R and Julia boundary is concise and proportionate. | Retain. |
| S08 | 44--45 | **Revise: abstract self-evaluation.** "This division makes ... explicit" tells the reader that the prose is clear rather than stating the consequence. | Replace with: "Readers can therefore distinguish calculations shared through Rust from those available only through a particular language interface." |
| S09 | 49--51 | **Pass.** The practical problem and heterogeneous tool setting are clear. | Retain. |
| S10 | 51--52 | **Pass.** Short, concrete transition. | Retain. |
| S11 | 52--54 | **Revise: noun pile.** Nine items in one sentence create a mechanical catalogue. | Replace with: "A result depends on strategy names, units, and parameter groups as well as its numerical values. Population, time horizon, implementation, study design, and provenance determine how that result should be interpreted." |
| S12 | 54--55 | **Pass.** It states the consequence of metadata loss without exaggeration. | Retain. |
| S13 | 57--58 | **Blocker: unsupported universal claim.** Validation and retained metadata vary across entry points; "malformed or inconsistent" does not identify what is checked. | Replace with: "The package provides inspectable decision and result objects, and each calculation validates the fields and array dimensions it requires." |
| S14 | 58--60 | **Pass.** The intended users and tasks are specific. | Retain. |
| S15 | 60--62 | **Revise: indirect and anthropomorphic.** | Replace with: "Decision descriptions accept analyst-supplied outcome and value units, so applications are not limited to health outcomes." |
| S16 | 62--63 | **Pass.** The domain of the example is clear. | Retain. |
| S17 | 67--68 | **Pass.** Restrained field-positioning claim with direct methodological sources. | Retain, after correcting the `rothery2020voi` metadata used elsewhere. |
| S18 | 69--70 | **Pass with citation maintenance.** The package scope is stated fairly. | Retain; verify the cited package version and date at the reviewed revision. |
| S19 | 70--71 | **Pass.** Concise and directly cited. | Retain. |
| S20 | 71--72 | **Pass.** Concise and directly cited. | Retain. |
| S21 | 72--74 | **Revise: citation coverage.** The application citation and method citation are useful, but a peer-reviewed assessment of web VOI tools would improve the comparison. | Replace after adding the source: "The Sheffield Accelerated Value of Information (SAVI) application provides a web interface for EVPI and regression-based EVPPI [@savi2025; @strong2014evppi; add the Tuffaha et al. web-tool review, DOI `10.1007/s40258-021-00662-4`]." |
| S22 | 74--76 | **Pass.** It avoids dismissing established alternatives. | Retain. |
| S23 | 78 | **Revise: vague comparative agent.** "Narrower settings" has no reference point until the next sentence. | Replace S23 and S24 with: "Python also has specialised alternatives: `value-of-information` analyses a single uncertain option observed through a noisy signal [@adamczewski2025valueofinformation], while `trd-cea-toolkit` embeds VOI in a disease-specific health-economic workflow [@mordaunt2025trdcea]." |
| S24 | 78--81 | **Revise with S23: moving software citation.** The comparison is useful, but the GitHub record needs an immutable tag or commit. | Use the combined replacement for S23 and cite a versioned release or commit for `value-of-information`. |
| S25 | 83--85 | **Pass.** The distinct requirement is stated without claiming methodological superiority. | Retain. |
| S26 | 85--87 | **Revise: hypothetical and vague requirement.** "The required language-neutral interface" assumes the conclusion. | Replace with: "The project therefore uses a Rust-owned interface rather than extending an R-centred workflow or maintaining separate numerical implementations in each language." |
| S27 | 87--89 | **Revise: awkward verb and abstract subject.** A choice does not "provide less depth". | Replace with: "The trade-off is less method-specific depth than established R packages and substantially narrower R and Julia interfaces than the Python interface." |
| S28 | 93--94 | **Pass.** The organising design decision is clear. | Retain. |
| S29 | 94--95 | **Pass.** The Rust responsibility is bounded by "selected". | Retain. |
| S30 | 95--96 | **Pass.** The Python responsibility is explained in user-facing terms. | Retain. |
| S31 | 97--98 | **Pass.** The R and Julia limitation is exact. | Retain. |
| S32 | 98--100 | **Revise: abstract subject and repeated "selected calculation".** | Replace with: "Sharing that calculation avoids three independent implementations, but each language still needs its own packaging and parity tests." |
| S33 | 102--104 | **Revise: governance-heavy noun chain.** | Replace with: "A method is labelled stable only when its inputs and outputs are specified, its results agree with an independent numerical reference, and tests cover repeatability and invalid inputs." |
| S34 | 104--106 | **Pass.** Approximation and optional-backend status are relevant distinctions. | Retain. |
| S35 | 106--107 | **Pass.** This is a necessary scientific limitation, not generic hedging. | Retain. |
| S36 | 107--109 | **Revise: noun pile and double-hyphen alias.** | Replace with: "The analytical EVSI calculation specifies a normal prior, a two-arm normal likelihood with equal allocation and known outcome variance, a linear net-benefit model, and the corresponding posterior update." |
| S37 | 109--110 | **Revise: internal label and incomplete criterion.** "Compatibility estimators" is unexplained, and absence of a declared likelihood alone is not the full readiness boundary. | Replace with: "Other callable EVSI estimators are labelled non-stable when they do not expose a complete, validated study model." |
| S38 | 112--114 | **Revise: compressed assurance catalogue.** "Software-assurance" is an unnecessary compound, and "implementation parity" is opaque to non-developers. | Replace with: "The repository tests calculations against analytical references, rejects invalid inputs, checks repeatability, compares shared results across implementations, and exercises supported installation environments." |
| S39 | 114--116 | **Revise: list cadence and implementation detail.** | Replace with: "These checks include unit, integration, generated-input, cross-implementation, mutation, clean-installation, and cross-platform tests." |
| S40 | 116--117 | **Pass.** The public release inventory matches the release-evidence manifest. | Retain only as a statement about `v1.0.0`, not the later working tree. |
| S41 | 117--119 | **Blocker: SBOM scope is too broad.** The current workflow inventories the Python distribution and installed Python environment, not the complete mixed-language workspace. | Replace with: "GitHub records build-provenance attestations. A separate workflow generates a CycloneDX inventory of the Python distribution and its installed dependencies." If a composed mixed-language SBOM is later produced, state that broader scope only after verification. |
| S42 | 123--124 | **Revise: inflated label and hyphen pile.** A synthetic example is not itself a research demonstration, and "cross-project research-workflow integration" is difficult to parse. | Replace with: "The repository contains one synthetic worked example and one same-author integration with another research workflow." |
| S43 | 124--126 | **Pass.** The decision and uncertain quantities are accessible. | Retain. |
| S44 | 126--127 | **Revise with S45: incomplete assumptions.** The health-effect prior is correct, but the cost distribution is omitted. | Replace S44 and S45 with: "The incremental health effect has a Normal prior with mean 0.06 and standard deviation 0.03; programme cost is sampled independently from a Normal distribution with mean 3,000 and standard deviation 650. The proposed study allocates 200 participants equally between two groups, assumes an individual outcome standard deviation of 1.0, and informs the health effect rather than programme cost." |
| S45 | 127--128 | **Blocker with S44: incomplete study model.** The existing sentence does not state which uncertainty the study informs. | Use the combined replacement for S44. |
| S46 | 130--131 | **Revise: awkward units and estimate status.** | Replace with: "When one health unit is valued at 50,000 value units, the programme is preferred in 49.24% of 10,000 fixed-seed simulations." |
| S47 | 131--132 | **Revise: estimators presented as exact quantities.** | Replace with: "In those simulations, estimated EVPI is 644 value units per person; regression-based EVPPI is 590 for health effect and 250 for programme cost." |
| S48 | 132--133 | **Revise: double-hyphen alias and missing sample-size link.** | Replace with: "For the proposed total sample of 200, the analytical model with a normal prior and normal likelihood gives an EVSI of 124 per person." |
| S49 | 133--135 | **Blocker: assumptions are not stated; interval wording hides evaluated points.** Insert the missing ENBS assumptions before this result: "ENBS uses 1,300 decision opportunities per year for ten years, discounted at 3%, and a study cost of 1.2 million plus 100 per participant. The delayed scenario excludes benefits for two years and applies 60% uptake thereafter." Then replace S49 with: "For immediate full uptake, ENBS is negative at 100 participants and positive at 200." |
| S50 | 135--136 | **Revise with S49: evaluated-point precision.** | Replace with: "With the delay and lower uptake, ENBS is negative at 800 participants and positive at 1,200." |
| S51 | 136--138 | **Revise: excessive strength and rule-of-six catalogue.** A synthetic example illustrates rather than shows a general causal result. | Replace with: "The synthetic example illustrates how study size and implementation assumptions affect estimated research value; it is not an estimate for a real trial." |
| S52 | 140--143 | **Pass.** The first caption sentence identifies all three panels without interpreting them beyond their contents. | Retain. |
| S53 | 142--143 | **Pass.** The synthetic-data boundary is explicit. | Retain. |
| S54 | 145--148 | **Revise: dense implementation nouns and mutable evidence.** The cited repository URL is not an immutable research-use record. | After recording an exact revision, replace with: "At revision `[commit or tag]`, the developer-led `vop_poc_nz` health-economic workflow tests its decision records, provenance fields, schema identifiers, and expected results against version 1.0.0 of the `voiage` compatibility contract [cite immutable revision]." |
| S55 | 148--149 | **Pass.** Same-author use and independent adoption are distinguished explicitly. | Retain. |
| S56 | 149--151 | **Revise: two claims joined by a semicolon.** | Replace with: "The package has been developed publicly since July 2025. Attributable non-author use has not yet been documented." |
| S57 | 155 | **Pass.** Artificial intelligence is expanded at first use, and the disclosure is direct. | Retain. |
| S58 | 155--158 | **Revise: mechanically long activity catalogue.** The named services and model-retention boundary should remain. | Replace with: "OpenAI Codex, using GPT-5-family models, and Google Jules, using service-managed models, assisted with repository analysis, code and test drafting, refactoring, documentation, workflow review, and manuscript editing." |
| S59 | 158--159 | **Pass.** It records a real disclosure limitation without speculative detail. | Retain. |
| S60 | 159--162 | **Revise: universal validation claim and list cadence.** | Replace with: "The author chose the research problem and architecture, reviewed this manuscript, and checked the reported code, references, and numerical results against repository tests and generated evidence." |
| S61 | 162--164 | **Pass.** Responsibility and non-authorship are explicit. | Retain. |
| S62 | 168 | **Pass.** Funding declaration is direct. | Retain. |
| S63 | 168--169 | **Pass.** Competing-interest declaration is direct. | Retain. |
| S64 | 173 | **Blocker: release-to-paper mismatch.** `v1.0.0` is public, but it predates the analytical EVSI and binding fixes described by the paper. | Replace only after an immutable reviewed release exists: "The software reviewed for this paper is `voiage` version `[version]` at commit `[commit]` [release citation]." |
| S65 | 173--174 | **Revise: unnecessary hyphen and incomplete provenance.** | Replace with: "The health example uses synthetic data, seed `20260723`, the script `scripts/generate_paper_health_example.py`, and the machine-readable outputs in `paper/data/`." |
| S66 | 174--177 | **Blocker if retained as review identity.** The SWHID is real for the recorded repository snapshot, but it is not evidence that the unreleased working-tree changes were archived. | After archiving the reviewed release, replace with: "Software Heritage preserves the reviewed release as `[SWHID]` [archive citation]." |
| S67 | 177--178 | **Blocker: prospective workflow language and unnamed revision.** A final manuscript cannot promise a future freeze, and "exact reviewed revision" does not identify the revision. | Replace with a factual sentence after release: "The release-evidence manifest at `[path or archival URL]` binds the reviewed version, commit, package assets, checksums, provenance attestations, SBOM scope, and Software Heritage identifier." |

### Inventory reconciliation

The manuscript contains 67 grammatical prose sentences when line wrapping is
removed. All 67 are classified above. The final availability sentence contains
two independently testable propositions, both assessed in S67. No prose was
omitted from review.

The non-sentence front matter is structurally complete: title, tags, author,
ORCID, affiliations, date, bibliography, and repository are present. The title
is descriptive rather than promotional. The tags mix concepts and language
names appropriately. The date must be regenerated if the reviewed release or
submission date changes. The headings use sentence case and match JOSS's
required content.

## Bibliography and citation audit

### Fail-closed citation defect

The `rothery2020voi` record at `paper.bib:24` incorrectly gives:

- `Murray, John F.` instead of **Murray, James F.**; and
- `Sanders Schmidler, Gary D.` instead of **Sanders Schmidler, Gillian D.**.

Replace the author field with:

> `Rothery, Claire and Strong, Mark and Koffijberg, Hendrik and Basu, Anirban and Ghabri, Salah and Knies, Saskia and Murray, James F. and Sanders Schmidler, Gillian D. and Steuten, Lotte and Fenwick, Elisabeth`

The DOI `10.1016/j.jval.2020.01.004`, title, journal, volume, issue, pages, and
year otherwise match the cited article. Because Authentext pattern A9 treats an
inaccurate citation as critical, the manuscript remains blocked even though
the error does not change the scientific argument.

### Other citation actions

1. Add the peer-reviewed review of online VOI tools by Tuffaha et al.,
   DOI `10.1007/s40258-021-00662-4`, to support and contextualise the SAVI
   comparison.
2. Replace moving GitHub URLs for `vop_poc_nz` and
   `value-of-information` with a release, tag, commit, or archived snapshot.
3. Update the `voiage` software citation to the exact release reviewed by the
   panel. Do not use the `v1.0.0` record to support features added later.
4. Update the Software Heritage citation to the snapshot that contains that
   reviewed release, or state precisely what the existing snapshot contains.
5. Keep bibliographic article titles unchanged even where they use American
   spelling; titles are technical literals.

Citation numbering will be determined by first appearance in the rendered JOSS
paper, not by physical order in `paper.bib`.

## Spelling, terminology, and accessibility

Australian/British spelling is consistent in the manuscript: "prioritising",
"modelling", "visualisation", "labelled", and "artefacts" are used correctly.
American spelling inside published article or package titles should not be
altered.

The abbreviations VOI, EVPI, EVPPI, EVSI, ENBS, SAVI, and AI are expanded on
first use. "CycloneDX" and "SBOM" do not appear in the manuscript, so no
expansion is currently required. If "SBOM" is introduced in the revised
availability sentence, first write "software bill of materials (SBOM)".

Terms that should be translated or removed for a non-developer audience:

- "compatibility estimators" → "other callable EVSI estimators";
- "implementation parity" → "the same result across implementations";
- "machine-readable inventory of software components" → name the CycloneDX
  inventory and its actual scope;
- "`normal--normal`" → "normal prior and normal likelihood";
- "compatibility contract" → explain what records and expected results are
  checked;
- "release-evidence manifest" → retain only when followed by what it binds.

The prose has useful sentence-length variation and no cluster of clipped
sentences. Its accessibility problem is lexical density, not sentence length.
The replacements above retain technical literals while moving implementation
details behind their research consequences.

## Fail-closed blockers

### Manuscript-owned blockers

1. **Release identity:** lines 42--43 describe functionality absent from the
   cited `v1.0.0` release, and lines 173--178 do not identify an immutable
   reviewed revision containing the described software.
2. **Worked-example assumptions:** lines 133--136 rely on population, horizon,
   discounting, research-cost, and delayed-uptake assumptions that the
   manuscript never states.
3. **SBOM scope:** lines 117--119 imply a general software-component inventory,
   while the inspected workflow produces a Python-environment CycloneDX
   inventory.
4. **Citation accuracy:** the Rothery bibliography record contains two
   incorrect author names.
5. **Availability tense:** "will be frozen before submission" is a plan, not
   availability evidence.

### Repository and external gates kept separate

- A new signed release, hosted checks, an official JOSS PDF build, and a
  revision-matched Software Heritage snapshot are repository or external
  readiness requirements. They become manuscript defects only when the paper
  attributes unreleased behaviour to an older release or claims evidence that
  does not exist.
- Attributable non-author installation or research use remains undocumented.
  The paper states that boundary honestly, so it is not an Authentext defect.
- The pending permanent arXiv identifier is external and need not appear in the
  JOSS paper until it exists.
- JOSS submission, editorial screening, review, acceptance, and DOI assignment
  are external outcomes and must not be described as completed.

## Prioritised revision actions

1. Publish or otherwise identify one immutable reviewed revision containing the
   EVSI, binding, packaging, manuscript, and evidence changes; then rewrite the
   availability section with its exact version, commit, manifest, and archive
   identifier.
2. Add the complete ENBS assumptions and rewrite the threshold results as signs
   observed at evaluated sample sizes.
3. Correct the Rothery author names and add the direct online-VOI-tool review.
4. Bound the structured-workflow, input-validation, and SBOM claims to what the
   inspected implementation and evidence actually cover.
5. Apply the exact replacements in S01--S67, especially the changes from
   "show" to "illustrate", from generic uncertainty reduction to decision
   improvement, and from internal engineering labels to research consequences.
6. Render the official JOSS PDF at the exact reviewed commit and perform a
   visual sentence and figure-caption check before the next panel round.

## Rubric score

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 164/180 | The scope is proportionate and same-author use is clearly bounded, but "research demonstration" overstates the synthetic example and the cited cross-project evidence is mutable. |
| Statement of need and audience | 114/120 | The transfer problem is clear; the package response overgeneralises metadata retention and input validation. |
| State of the field and build-versus-contribute case | 112/130 | Comparisons are fair, but a direct web-tool review and immutable software citations are missing, and the separate-package rationale is partly hypothetical. |
| Scientific and numerical accuracy | 132/150 | Point estimates match generated outputs, but the study-value assumptions are incomplete in the paper and simulation estimates need clearer labels. |
| Software design and research relevance | 94/100 | The architecture is understandable; maturity and assurance language becomes repository-facing and noun-heavy. |
| Reproducibility, packaging, documentation, and tests | 78/100 | The paper names real release assets and evidence, but the described code, cited release, SBOM scope, and future reviewed revision are not reconciled. |
| Research-impact statement | 66/80 | The example and same-author workflow are concrete and candidly bounded; neither establishes independent adoption, and the first is labelled too strongly. |
| Structure, metadata, and JOSS format | 58/60 | Required sections and metadata are present; availability remains prospective. |
| Clarity, accessibility, and sentence quality | 52/55 | Tone and spelling are restrained; remaining deductions are for catalogue cadence, abstract demonstrative subjects, double-hyphen compounds, and a few dense engineering phrases. |
| Citations, provenance, declarations, and AI disclosure | 14/25 | Declarations and disclosure are complete, but the Rothery record is inaccurate, software references are mutable, and release/archive provenance does not identify the reviewed code. |
| **Total** | **884/1000** | **Fail-closed major revision.** |

## Round 3 acceptance conditions for this role

This sentence-editor role can score above 995 only when:

- every replacement is reconciled against the scientific and software
  reviewers so that copy-editing does not alter meaning;
- the manuscript names the exact reviewed release and archive rather than a
  future action;
- all assumptions necessary to interpret the worked example appear in the
  paper;
- every citation record resolves with accurate metadata and software citations
  are immutable;
- the official rendered PDF contains no line-wrap, caption, reference, or
  symbol defect;
- a fresh sentence inventory has no unresolved finding or blocker; and
- the next review is performed against one committed, hosted, immutable
  revision rather than a dirty working tree.
