# Authentext sentence-level review

**Recommendation:** major revision before JOSS submission.
**Score:** **771/1000**.
**Revision reviewed:** `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`.

The manuscript is mechanically clean: the repository JOSS validator passes, and Vale reports no findings. Its main weaknesses are substantive rather than grammatical. In particular, it overstates the public EVSI capability, describes an interoperability demonstration as a research use, gives an incomplete comparison with related software, and sometimes substitutes implementation terminology for an explanation of research value.

JOSS asks for a jargon-light opening, a defensible comparison with commonly used software, design choices explained through their research consequences, and specific evidence of realised or credible near-term research impact. [JOSS paper guidance](https://joss.readthedocs.io/en/latest/paper.html), [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)

## Overall argument

The paper’s most defensible central claim is:

> `voiage` provides an explicit decision-and-result contract for carrying labelled VOI analyses across language boundaries, while distinguishing calculations shared through Rust from modelling and reporting performed by individual language interfaces.

That argument is present, but obscured by feature catalogues and low-level terminology. The revised sequence should be:

1. VOI decisions depend on more than formulas: strategy labels, uncertainty draws, population assumptions, proposed-study assumptions, units, and provenance affect interpretation.
2. Moving an analysis between tools or languages can lose or alter that information.
3. Existing packages provide mature specialist workflows.
4. `voiage` addresses a different requirement by combining explicit decision/result contracts with selected shared Rust calculations.
5. The health example shows why study size, delay, uptake, population, and cost assumptions matter.
6. Python currently has the broadest interface; R and Julia share EVPI only.
7. The repository supplies strong software-assurance evidence, but independent adoption is not yet established.

The present paper changes abstraction level too abruptly: applied decision problem → feature catalogue → foreign-function architecture → research impact. That is Authentext reasoning pattern R4. The software design section should connect each engineering decision to an applied consequence before naming its implementation.

## Complete sentence inventory

| # | Lines | Assessment | Proposed replacement or action |
|---:|---:|---|---|
| 1 | 30–31 | **Finding:** conflates perfect information with collecting data. | “Value of Information (VOI) analysis estimates the expected improvement in a decision outcome from resolving uncertainty under a specified information scenario.” |
| 2 | 31–34 | **Finding:** broad domain claim needs direct evidence. | “Perfect-information measures describe hypothetical resolution of all or selected uncertainty, whereas sample-information measures evaluate a specified proposed study. VOI has applications in health economics, trial design, public policy, and environmental management [add direct sources].” |
| 3 | 34–35 | **Finding:** “computes and communicates” is vague. | “`voiage` takes strategy-labelled outputs from probabilistic decision models and calculates measures used to assess uncertainty and proposed research.” |
| 4 | 35–39 | **Blocker:** unqualified EVSI claim is contradicted by the public implementation. | “The package provides EVPI, regression-based EVPPI, ENBS, cost-effectiveness acceptability frontiers, dominance analysis, and diagnostics. The accompanying health example evaluates a proposed study with a separately specified normal–normal EVSI model; other EVSI estimators require method-specific validation.” |
| 5 | 41–43 | **Finding:** implementation jargon, American “serialization”, and an abstract list. | “Version 1.0.0 uses Rust to define shared input and result types and to perform selected calculations used by more than one language.” |
| 6 | 43–44 | **PASS:** accurate capability boundary. | Retain, optionally changing “surface” to “interface”. |
| 7 | 44–46 | **Finding:** technically accurate but cumbersome. | “The current R and Julia packages call the same Rust implementation for EVPI, but expose fewer methods than Python and are not yet available from their language registries.” |
| 8 | 46–48 | **Finding:** “convenience function” is unexplained and diminishes meaningful language-specific work. | “This distinction tells users which results come from shared Rust calculations and which are produced by one language interface.” |
| 9 | 52–54 | **Finding:** useful premise, but “additional research” and “fragmented” are generic. | “VOI can inform whether the expected benefit of a proposed study exceeds its cost, but analysts conduct these calculations through several specialist packages, web tools, languages, and model-output formats.” |
| 10 | 54–56 | **Finding:** uncited empirical assertion. | “Moving an analysis between these environments requires analysts to preserve strategy labels, uncertainty draws, parameter groupings, population assumptions, and the distinction between perfect and sample information.” |
| 11 | 56–58 | **PASS in substance:** concrete risk, but it follows better from the proposed sentence 10. | Retain after changing “It also creates a risk” to “If that information is lost”. |
| 12 | 60–62 | **Finding:** “one clear description” and “careful data checks” are vague self-evaluations. | “`voiage` records strategy-labelled uncertainty draws, population and study assumptions, validation findings, warnings, and provenance in results that can be saved and inspected.” |
| 13 | 62–65 | **PASS:** audience and research tasks are clear. | Retain. |
| 14 | 65–68 | **Blocker:** implies wider R/Julia parity than exists. | “Python provides the broadest interface. The current R and Julia packages use the same versioned Rust calculation for EVPI; wider VOI functionality remains narrower than Python and is not claimed to have complete cross-language parity.” |
| 15 | 72–73 | **PASS:** concise and appropriately cited. | Retain. |
| 16 | 74–76 | **Finding:** fair but incomplete field comparison; reconcile the `voi` citation year and add `dampack`. | “The R packages `voi`, `BCEA`, and `dampack` provide established VOI and cost-effectiveness workflows through different combinations of estimation, diagnostics, and graphical reporting [citations].” |
| 17 | 76–78 | **Blocker:** `strong2014evppi` supports the regression method, not the SAVI software claim; SAVI is not expanded. | “The Sheffield Accelerated Value of Information (SAVI) application provides a web interface for regression-based EVPPI [add direct SAVI source; retain Strong et al. for the method].” |
| 18 | 78–80 | **PASS:** restrained and fair to alternatives. | Retain. |
| 19 | 82 | **Finding:** negative framing expends space without advancing the comparison. | “These tools and `voiage` address different requirements.” |
| 20 | 82–85 | **Finding:** distinctive scope is abstract and “cross-domain” is prospective. | “`voiage` was developed for a versioned decision-and-result contract that can be exercised from Python, R, and Julia while retaining labels, provenance, and maturity information.” |
| 21 | 86–88 | **Blocker:** speculates about the purpose and intentions of other packages. | “An extension to an R-centred package would not by itself provide the required language-neutral binary boundary, while independent implementations would require separate parity validation.” |
| 22 | 88–90 | **Finding:** important limitation weakened by “stable surface”. | “The trade-off is less methodological depth than specialist R tools and substantially narrower R and Julia interfaces.” |
| 23 | 94 | **PASS:** clear architectural principle. | Retain. |
| 24 | 95–97 | **Finding:** PyO3, “stable reductions”, and dependency ownership are too low-level for the paper. | “Rust contains the shared types and selected calculations. Native language bindings expose them without adding plotting or language-specific data-processing dependencies to the numerical library.” |
| 25 | 97–99 | **Finding:** “promotion criteria” is internal governance language. | “Python handles labelled data, user-defined models, reporting, and methods that are not yet shared across languages.” |
| 26 | 99–100 | **PASS:** exact and bounded. | Retain. |
| 27 | 102–103 | **Finding:** “larger repository” is not the meaningful research trade-off. | “This architecture requires maintained interfaces and parity tests, but lets selected calculations use the same implementation across languages.” |
| 28 | 103–105 | **Finding:** reviewer-centred rather than research-centred, although relevant to JOSS. | “Reviewers can test the Rust calculations independently and install the Python wheel in a clean environment, reducing ambiguity about which layer produced a result.” |
| 29 | 105–106 | **PASS:** useful packaging information stated economically. | Retain. |
| 30 | 106–108 | **Finding:** maturity labels are sensible, but the current “stable” EVSI status is not scientifically supported. | “The documentation labels methods as stable, developing, or experimental according to stated maturity criteria. These labels do not establish that an estimator is suitable for a particular decision problem.” |
| 31 | 110–113 | **Blocker:** “Correctness evidence” overstates what software tests establish. | “Software-assurance evidence includes analytical fixtures, unit and integration tests, property-based and differential tests, mutation testing, Rust fuzzing and Miri, cross-language conformance fixtures, clean installations, and cross-platform continuous integration. Analytical and simulation-reference tests are used where a trustworthy comparator exists; software tests do not by themselves establish the suitability of an estimator.” |
| 32 | 113–114 | **Blocker:** the GitHub release does not contain an SBOM as a release asset. | “The v1.0.0 GitHub release contains the source distribution, platform wheels, and checksums. Its workflow creates provenance attestations, while a separate assurance workflow retains the software bill of materials.” |
| 33 | 118 | **Blocker:** misclassifies an interoperability demonstration as a research use. | “The repository contains one synthetic research example and one cross-project interoperability demonstration.” |
| 34 | 118–121 | **Blocker:** the study-size EVSI calculation is not run through the public `voiage.evsi` implementation. | “The synthetic health example uses `voiage` for EVPI, EVPPI, and ENBS, together with a script-specific normal–normal EVSI calculation for a proposed two-arm study.” |
| 35 | 121–122 | **Finding:** reports reproducibility mechanics but no result. | “At 50,000 per quality-adjusted life-year, the programme is preferred in 49.24% of draws; EVPI is 644 per person, and EVPPI is 590 for health effect and 250 for programme cost. Immediate full-uptake ENBS becomes positive at 200 participants, whereas delayed 60% uptake requires 1,200 participants in the evaluated design.” |
| 36 | 123–125 | **Finding:** valid evidence but noun-heavy. | “A separate integration contract with `vop_poc_nz` transfers decision records, provenance, schema fingerprints, and expected numerical results between the two repositories.” |
| 37 | 125–126 | **PASS:** correctly denies independent adoption. | Retain, preferably “These are developer-created reproducible materials, not evidence of independent adoption.” |
| 38 | 128 | **PASS:** specific development-history fact. | Retain. |
| 39 | 128–131 | **Finding:** mostly aspirational and repeats the external-adoption boundary. | “The release, documentation, and fixed-seed example provide material that other researchers can inspect and reuse. Attributable non-author installation or research use has not yet been documented.” |
| 40 | 135–138 | **Finding:** required disclosure is complete, but “AI” is not expanded and the list is mechanically long. | “Generative artificial intelligence (AI) tools assisted with this work: OpenAI Codex using GPT-5-family models and Google Jules using service-managed models were used for repository analysis, drafting code and tests, refactoring, documentation, security and release review, and manuscript editing.” |
| 41 | 138–140 | **PASS:** necessary limitation on historical model records. | Retain. |
| 42 | 140–143 | **PASS:** records human validation without overstating independence. | Retain. |
| 43 | 143–145 | **PASS:** preserves responsibility, authorship, and disclosure requirements. | Retain. |
| 44 | 145–146 | **PASS:** clear JOSS interaction boundary. | Retain, optionally replacing “JOSS” with “the journal’s” to avoid an unexpanded acronym. |
| 45 | 150–152 | **Finding:** generic acknowledgement and rule-of-four catalogue; names no attributable contribution. | Omit unless a contributor can be named, or use: “The author thanks [name] for [specific contribution].” |
| 46 | 152 | **Finding:** “is declared” is indirect. | “This work received no external funding.” |
| 47 | 152–153 | **PASS:** direct conflict declaration. | Retain. |
| 48 | 157–159 | **PASS in substance, structural finding:** availability prose should not sit under “References”. | Move to “Software and data availability”: “`voiage` version 1.0.0 is available from the public release [@voiage2026], and the reviewed repository snapshot is preserved by Software Heritage [@voiage_software_heritage].” |
| 49 | 159–160 | **Finding:** peripheral metadata claim that does not advance the paper. | Remove the sentence and `smith2016software`, unless software-citation metadata becomes part of the argument. |

## Cohesion and Authentext findings

### Paragraph and section structure

- The Summary moves from an accessible definition to Rust terminology too quickly.
- The Statement of need repeats the Summary’s package description instead of developing the concrete transfer problem.
- The State of the field is concise, but incomplete and partly unsupported.
- Software design inventories technology before explaining applied consequences.
- Research impact describes how artefacts can be rerun but says too little about what they show.
- Availability information is incorrectly placed under References.

A separate conclusion is not required by JOSS and would probably waste the word allowance. The final research-impact or availability paragraph can close the argument.

### Spelling and terminology

Australian/British spelling is mostly consistent:

- Correct: “modelling”, “labelled”, “behaviour”, “specialised”, “artefacts”, “Acknowledgements”.
- Change “serialization” to “serialisation”.
- Preserve “Modeling” in the Ades article title because bibliographic titles are technical literals.
- “Programme” should remain the spelling used for the intervention.

### Acronyms and jargon

Properly expanded at first use:

- VOI
- EVPI
- EVPPI
- EVSI
- ENBS

Not adequately introduced:

- AI → “artificial intelligence (AI)”
- SAVI → “Sheffield Accelerated Value of Information (SAVI)”
- JOSS → avoid or expand as “Journal of Open Source Software (JOSS)”
- PyO3 → remove from the paper or explain outside it
- “binding-independent”, “stable reductions”, “surface”, “promotion criteria”, “schema fingerprints”, and “versioned C boundary” require translation into user consequences

### Cadence and AI-pattern clusters

There is little promotional or hyperbolic language. No “groundbreaking”, “novel”, “comprehensive”, “crucial”, “pivotal”, or similar language appears.

The stronger Authentext signals are structural:

- repeated catalogue sentences;
- evenly paced, medium-length declaratives;
- clusters of abstract nouns;
- several rule-of-three or rule-of-four sequences;
- vague self-evaluations such as “one clear description” and “careful data checks”;
- shifts between applied-research language and internal engineering vocabulary;
- correctness and impact claims that are stronger than their evidence.

These clusters matter more than any isolated word. There is no em-dash, rhetorical-hook, fake quotation, staccato-drama, or excessive-hedging problem.

### Technical literal preservation

Future edits should preserve exactly:

- inline identifiers such as `voiage`, `voi`, `BCEA`, and `vop_poc_nz`;
- version `1.0.0`;
- ORCID, URLs, DOI values, and the Software Heritage identifier;
- article titles in `paper.bib`;
- language and project names such as Rust, Python, R, Julia, OpenAI Codex, Google Jules, and GPT-5;
- the substantive AI disclosure boundaries.

## Bibliography findings

The seven current records are syntactically usable and all are cited, but the bibliography is not yet sufficient for the revised argument.

Required changes:

- Add direct methodological guidance distinguishing EVPI, EVPPI, EVSI, and ENBS, such as Rothery et al. (2020), DOI `10.1016/j.jval.2020.01.004`.
- Add `dampack`.
- Add a direct source for SAVI; retain Strong et al. for regression-based EVPPI.
- Consider the directly relevant Python package `value-of-information`.
- Add direct sources for non-health applications if those domains remain in the Summary.
- Reconcile the intended `voi` package version and year.
- Remove `smith2016software` if its only purpose is the peripheral metadata sentence.

Reference numbering will be generated in order of first citation by the JOSS rendering toolchain; the physical order of entries in `paper.bib` does not determine citation numbering.

## Rubric score

| Dimension | Score | Deduction |
|---|---:|---|
| Scope, significance, and research use | 144/180 | Research and interoperability evidence are conflated; no independent use |
| Statement of need and audience | 102/120 | Gap is generic; concrete transfer consequences appear too late |
| State of the field and build-versus-contribute case | 96/130 | Missing comparators, indirect SAVI citation, speculative rationale |
| Scientific and numerical accuracy | 86/150 | Public EVSI claim and example attribution are materially misleading |
| Software design and research relevance | 84/100 | Excessive implementation vocabulary and weak applied consequences |
| Reproducibility, packaging, documentation, and tests | 89/100 | SBOM/release boundary is inaccurate; scientific-reference validation is incomplete |
| Research-impact statement | 48/80 | Developer example and interoperability contract are overstated as research uses |
| Structure, metadata, and JOSS format | 57/60 | Availability prose is under References |
| Clarity, accessibility, and sentence quality | 47/55 | Jargon, abstract lists, uniform cadence, one spelling inconsistency |
| Citations, provenance, declarations, and AI disclosure | 18/25 | Missing or mismatched sources; AI acronym not expanded |
| **Total** | **771/1000** | **Major revision** |

The fail-closed cap applies because material scientific, impact, release, and state-of-field claims remain unsupported or contradicted by repository evidence.

## External and repository gates

These do not disappear if the manuscript is copy-edited:

- **Repository-owned scientific blocker:** the valid normal–normal EVSI analysis is paper-specific, while several public EVSI estimators do not encode a defensible proposed-study sampling model. This requires implementation and validation work or explicit demotion of the affected estimators.
- **External evidence gate:** [issue #471](https://github.com/edithatogo/voiage/issues/471) remains open with only the maintainer’s request for validation and no attributable non-author result.
- **External editorial gate:** JOSS screening, review, and acceptance can only be decided by its human editorial process.
- **Author sequencing preference:** the permanent arXiv identifier is still relevant to the intended submission sequence, but it is not itself a JOSS requirement.
- **Later archival gate:** the DOI for the exact reviewed release is normally prepared at the acceptance stage.

No files were edited.

<oai-mem-citation>
<citation_entries>
MEMORY.md:1340-1346|note=[Used prior manuscript review context while keeping publication claims separate from validation]
MEMORY.md:1436-1442|note=[Used the established arXiv before JOSS sequencing and human submission boundary]
</citation_entries>
<rollout_ids>
</rollout_ids>
</oai-mem-citation>
