# Simulated JOSS pre-review screening report

**Repository:** `edithatogo/voiage`
**Manuscript:** [paper.md](/Volumes/PortableSSD/GitHub/voiage/paper.md:1)
**Review date:** 24 July 2026
**Role simulated:** JOSS Associate Editor-in-Chief
**Status:** Internal AI-assisted quality review only. This is not a JOSS editorial decision.

## 1. Scope and review standard

This review applies the repository’s ten-dimension [JOSS panel rubric](/Volumes/PortableSSD/GitHub/voiage/docs/reviews/joss-panel/rubric.md:1) and the current official:

- [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html)
- [JOSS paper requirements](https://joss.readthedocs.io/en/latest/paper.html)
- [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
- [JOSS editor guidance](https://joss.readthedocs.io/en/latest/editing.html)

The review distinguishes:

1. Manuscript and repository quality.
2. Claims that are not adequately supported.
3. Correctable pre-review blockers.
4. External or later-stage gates, including issue #471, editorial assignment, review and archival DOI creation.

The manuscript was reviewed sentence by sentence. Claims were treated as unverified unless supported by the repository, release artifacts, authoritative registries, cited literature or reproducible evidence.

## 2. Evidence inspected

### Manuscript and submission material

- [paper.md](/Volumes/PortableSSD/GitHub/voiage/paper.md:1)
- [paper.bib](/Volumes/PortableSSD/GitHub/voiage/paper.bib:1)
- [JOSS submission readiness](/Volumes/PortableSSD/GitHub/voiage/docs/release/joss-submission-readiness.md:1)
- [Panel rubric](/Volumes/PortableSSD/GitHub/voiage/docs/reviews/joss-panel/rubric.md:1)
- Rendered JOSS PDF from the successful hosted workflow
- Manuscript metadata, section structure, word count and references

### Software and research evidence

- Python package, Rust core and foreign-function interface
- R and Julia source bindings
- Public API and diagnostic contracts
- Tests, fixtures and cross-language conformance checks
- Synthetic health example, fixed seeds, generated values and figure
- `vop_poc_nz` integration contract and provenance records
- Documentation, installation instructions and contributor material
- Git history, tags, releases, contributors and public development period
- v1.0.0 release assets, checksums and GitHub provenance attestations
- Software Heritage snapshot and tag inclusion
- SBOM workflow and its relationship to the GitHub release
- GitHub issue #471 and its current evidence
- Current CRAN `voi` metadata and other state-of-the-field sources

### Verification results

- `scripts/validate_joss.py`: passed.
- `tox -e joss`: seven tests passed, with two non-fatal xarray warnings.
- Hosted JOSS build: passed.
- Rendered manuscript: three pages, approximately 1,005 body words, with no obvious rendering failure.
- Python 1.0.0 clean installation: package installation succeeded.
- Import and quick-start calculation succeeded after constraining native numerical-library thread counts. This is an environment-specific review risk, not sufficient evidence of a general package defect.
- R and Julia do route EVPI matrix calculations to Rust, but do not provide feature parity with the Python interface.
- GitHub provenance attestations cover released package artifacts.
- The GitHub release does not contain an SBOM asset. An SBOM is generated separately by a workflow.

## 3. Overall finding

`voiage` is clearly research software and has an obvious application to value-of-information analysis. It has a public repository, an OSI-approved licence, more than six months of visible development, substantive tests, documentation, releases and evidence of iterative engineering. Its paper meets the basic JOSS length and section requirements.

The submission is nevertheless not ready to pass pre-review unchanged.

The largest problem is not software engineering quality. It is the precision with which the paper represents the available evidence:

- Author-created demonstrations are described as “two concrete developer-led research uses.”
- The release is said to include an SBOM when the SBOM is actually a separate workflow artifact.
- Cross-language consistency is described more broadly than the current EVPI-only shared implementation supports.
- The state-of-the-field section does not yet establish a sufficiently complete, sourced build-versus-contribute case.
- One bibliography record has incorrect authorship metadata.
- Some impact language is aspirational, which JOSS explicitly does not accept as a substitute for demonstrated impact.

These are correctable manuscript problems. Issue #471 is different: it concerns attributable non-author installation or research-use evidence. That evidence is not presently available. It should not be misrepresented as a paper-writing defect, but it creates material screening risk for a single-author submission without demonstrated external engagement.

## 4. Pre-review blockers

### B1. Research-use evidence is overstated

At [L118–126](/Volumes/PortableSSD/GitHub/voiage/paper.md:118), the manuscript calls the synthetic worked example and same-author repository integration “two concrete developer-led research uses.”

The evidence supports:

- A reproducible synthetic demonstration.
- A same-author integration contract with another research repository.
- Research readiness and intended research relevance.

It does not establish two completed uses of the software in research. This is a material claim-to-evidence mismatch.

**Required action:** Call these “research-readiness materials,” “worked demonstrations” or “developer-led integration evidence.” Reserve “research use” for attributable evidence that the software was actually employed in a research activity.

### B2. The release/SBOM claim is inaccurate

At [L110–114](/Volumes/PortableSSD/GitHub/voiage/paper.md:110), the paper says that the v1.0.0 release includes an SBOM.

The release contains package artifacts, checksums and verifiable provenance attestations. The SBOM is generated by a separate workflow and is not an asset attached to the release.

**Required action:** Distinguish release assets from separately generated CI evidence.

### B3. State of the field is incomplete

At [L72–90](/Volumes/PortableSSD/GitHub/voiage/paper.md:72), the manuscript discusses `voi`, BCEA and SAVI, but:

- The comparison is predominantly health-economic despite a broader cross-domain claim.
- Directly relevant alternatives and adjacent tools are not systematically considered.
- The build-versus-contribute justification speculates that extending another package would change its purpose.
- The actual current distinction—a contract-centred package with a shared Rust EVPI implementation—is broader in the prose than in the released binding coverage.

JOSS expects a clear comparison with commonly used alternatives and a defensible explanation of why a new package was warranted.

**Required action:** Expand or narrow the comparison, source every material comparison, and ground the build decision in concrete architectural and domain requirements rather than speculation about maintainers’ intentions.

### B4. Cross-language claims exceed current coverage

The statements at [L65–68](/Volumes/PortableSSD/GitHub/voiage/paper.md:65), [L82–85](/Volumes/PortableSSD/GitHub/voiage/paper.md:82) and [L102–103](/Volumes/PortableSSD/GitHub/voiage/paper.md:102) suggest a generally shared calculation boundary and consistent behaviour across languages.

The verified common route is presently EVPI for net-benefit matrices. Python exposes substantially more functionality. R and Julia are not full-equivalence interfaces.

**Required action:** Qualify every cross-language claim by method and interface maturity.

### B5. Citation metadata is unresolved

The `voi` bibliography entry lists Gianluca Baio as an author. Current authoritative CRAN metadata identifies Christopher Jackson and Anna Heath as authors, with Baio credited as a contributor for BCEA-derived code. See the [current CRAN `voi` manual](https://cran.r-project.org/web/packages/voi/voi.pdf).

The SAVI statement also relies principally on a methodological article rather than a direct citation to the software or service being described.

**Required action:** Correct the `voi` authorship and add an authoritative SAVI software/service citation.

## 5. External and later-stage gates

These should not be conflated with manuscript blockers.

| Gate | Current classification | Consequence |
|---|---|---|
| Issue #471: attributable non-author installation or research-use evidence | Open and unfulfilled | Material screening risk, especially for a single-author project; not itself a sentence-level paper defect |
| arXiv permanent identifier | Awaiting external processing | Author-preferred sequencing only; JOSS allows preprints before or during submission |
| JOSS editorial assignment | External | Cannot be satisfied inside the repository |
| Peer review and acceptance | External | Begins only after screening |
| Archival release DOI | Acceptance-stage requirement | Should be created for the reviewed release when requested by JOSS, not presented as a pre-review requirement |
| Independent adoption | Not verified | Must not be claimed; lack of it does not automatically make the software out of scope, but weakens impact and engagement evidence |

Official JOSS guidance treats community engagement and use beyond the authors as strong evidence. The reviewer criteria are particularly sceptical of single-author software with neither outside use nor collaborative engagement. Issue #471 therefore represents genuine eligibility risk, even though JOSS does not state a universal rule that every submission must already have an independent user.

## 6. Dimension scores

| Dimension | Maximum | Score | Deductions |
|---|---:|---:|---|
| 1. Scope, significance and research use | 180 | **125** | −30 for describing demonstrations as research uses; −15 for no attributable external use; −10 for a broader domain scope than the presented evidence establishes |
| 2. Statement of need and audience | 120 | **101** | −10 for largely uncited fragmentation claims; −5 for vague descriptions such as “one clear description” and “careful checks”; −4 for insufficiently bounded cross-domain claims |
| 3. State of the field and build-versus-contribute case | 130 | **82** | −20 for incomplete comparator coverage; −12 for a health-focused comparison supporting a cross-domain proposition; −10 for speculative claims about changing other packages’ purposes; −6 for underexplained distinctions from existing tools |
| 4. Scientific and numerical accuracy | 150 | **135** | −10 for overbroad cross-language consistency claims; −5 for insufficient separation of the EVPI shared core from wider interface functionality |
| 5. Software design and research relevance | 100 | **88** | −7 for overstating the breadth of the common Rust route; −5 for language that obscures the intentionally unequal current binding surfaces |
| 6. Reproducibility, packaging, documentation and tests | 100 | **86** | −6 for the inaccurate SBOM release statement; −4 for constrained-environment import sensitivity; −4 for divergence between release-tag metadata and current repository metadata |
| 7. Research impact | 80 | **44** | −18 for misclassifying demonstrations as research uses; −8 for uncited same-author integration evidence; −6 for aspirational future-impact language; −4 for no independent use evidence |
| 8. Structure, metadata and JOSS format | 60 | **57** | −3 for manuscript date drift; required sections, length and rendering otherwise pass |
| 9. Clarity, accessibility and sentence quality | 55 | **47** | −4 for specialist implementation language in the Summary; −2 for dense abstract nouns; −2 for sentences carrying too many separate propositions |
| 10. Citations, provenance, declarations and AI disclosure | 25 | **15** | −4 for incorrect `voi` authorship; −2 for inadequate direct support for SAVI; −2 for uncited `vop_poc_nz` evidence; −2 for a software-citation sentence that functions as citation padding |

**Total: 780/1000**

The raw total is below the rubric’s pass threshold. Independently, the material unsupported or inaccurate claims trigger the rubric’s fail-closed cap, so the paper could not receive more than 950 until those matters were resolved.

## 7. Sentence-level audit

Every substantive sentence in the manuscript is covered below.

| Sentence | Lines | Finding | Required treatment |
|---:|---|---|---|
| 1 | [30–31](/Volumes/PortableSSD/GitHub/voiage/paper.md:30) | Clear definition of value-of-information analysis. | Retain; a foundational citation would strengthen it. |
| 2 | [31–34](/Volumes/PortableSSD/GitHub/voiage/paper.md:31) | Broad policy, healthcare and business applicability is plausible but unsourced. | Cite cross-domain applications or narrow the wording. |
| 3 | [34–35](/Volumes/PortableSSD/GitHub/voiage/paper.md:34) | Accurate high-level description. | Retain. |
| 4 | [35–39](/Volumes/PortableSSD/GitHub/voiage/paper.md:35) | Feature inventory is substantially repository-backed. | Retain, but avoid allowing the inventory to dominate the Summary. |
| 5 | [41–43](/Volumes/PortableSSD/GitHub/voiage/paper.md:41) | Technically accurate in direction, but too implementation-heavy for a non-specialist Summary. | Rewrite in plain language. |
| 6 | [43–44](/Volumes/PortableSSD/GitHub/voiage/paper.md:43) | Correctly identifies Python as the wider interface. | Retain. |
| 7 | [44–46](/Volumes/PortableSSD/GitHub/voiage/paper.md:44) | R/Julia boundary is too easily read as broad parity. | State explicitly that they presently share EVPI only. |
| 8 | [46–48](/Volumes/PortableSSD/GitHub/voiage/paper.md:46) | “Versioned interface” and “language registry” are specialist concepts with little Summary value. | Remove or simplify. |
| 9 | [52–54](/Volumes/PortableSSD/GitHub/voiage/paper.md:52) | Fragmentation claim is plausible but general. | Add literature or concrete examples. |
| 10 | [54–56](/Volumes/PortableSSD/GitHub/voiage/paper.md:54) | Consequence claim needs evidence or qualification. | Use “can make” and cite an appropriate source. |
| 11 | [56–58](/Volumes/PortableSSD/GitHub/voiage/paper.md:56) | Interpretation risk is credible but insufficiently sourced. | Cite or hedge. |
| 12 | [60–62](/Volumes/PortableSSD/GitHub/voiage/paper.md:60) | “One clear description” and “careful checks” are vague. | Name the schemas, validation and diagnostic outputs. |
| 13 | [62–65](/Volumes/PortableSSD/GitHub/voiage/paper.md:62) | Audience is specific and appropriate. | Retain. |
| 14 | [65–68](/Volumes/PortableSSD/GitHub/voiage/paper.md:65) | Materially overbroad: only the EVPI path is shared across current bindings. | Replace. |
| 15 | [72–73](/Volumes/PortableSSD/GitHub/voiage/paper.md:72) | Fair introduction to an established field. | Retain. |
| 16 | [74–76](/Volumes/PortableSSD/GitHub/voiage/paper.md:74) | General comparison is supportable, but `voi` metadata is wrong. | Correct bibliography and ensure feature comparison matches current releases. |
| 17 | [76–78](/Volumes/PortableSSD/GitHub/voiage/paper.md:76) | SAVI description requires a direct software/service source. | Cite [SAVI](https://savi.shef.ac.uk/SAVI/) directly. |
| 18 | [78–80](/Volumes/PortableSSD/GitHub/voiage/paper.md:78) | Appropriately acknowledges the value of existing tools. | Retain. |
| 19 | [82](/Volumes/PortableSSD/GitHub/voiage/paper.md:82) | Useful non-displacement statement. | Retain. |
| 20 | [82–85](/Volumes/PortableSSD/GitHub/voiage/paper.md:82) | “One Rust calculation boundary” is broader than verified shared coverage. | Specify the common EVPI implementation. |
| 21 | [86–88](/Volumes/PortableSSD/GitHub/voiage/paper.md:86) | Speculates about how extending other packages would affect their purpose. | Replace with concrete unmet requirements and contribution constraints. |
| 22 | [88–90](/Volumes/PortableSSD/GitHub/voiage/paper.md:88) | Balanced acknowledgement of trade-offs. | Retain, with specific examples. |
| 23 | [94](/Volumes/PortableSSD/GitHub/voiage/paper.md:94) | Accurate architectural separation. | Retain. |
| 24 | [95–97](/Volumes/PortableSSD/GitHub/voiage/paper.md:95) | Repository-backed adapter and dependency description. | Retain. |
| 25 | [97–99](/Volumes/PortableSSD/GitHub/voiage/paper.md:97) | Accurate qualification of Python’s broader functionality. | Retain. |
| 26 | [99–100](/Volumes/PortableSSD/GitHub/voiage/paper.md:99) | “Released EVPI path” is ambiguous for interfaces not independently published in their registries. | Use “included source interfaces” or identify exact released artifacts. |
| 27 | [102–103](/Volumes/PortableSSD/GitHub/voiage/paper.md:102) | Cross-language consistency claim is too broad. | Limit to tested EVPI behaviour. |
| 28 | [103–105](/Volumes/PortableSSD/GitHub/voiage/paper.md:103) | Reviewer-visible testing rationale is sound. | Retain. |
| 29 | [105–106](/Volumes/PortableSSD/GitHub/voiage/paper.md:105) | Optional-dependency statement is supported. | Retain. |
| 30 | [106–108](/Volumes/PortableSSD/GitHub/voiage/paper.md:106) | Maturity-label statement is useful and evidence-backed. | Retain. |
| 31 | [110–113](/Volumes/PortableSSD/GitHub/voiage/paper.md:110) | Test catalogue broadly reflects repository automation, but “CI covers” is safer than implying all evidence belonged to the release. | Revise. |
| 32 | [113–114](/Volumes/PortableSSD/GitHub/voiage/paper.md:113) | Materially inaccurate regarding the SBOM being included in the release. | Replace. |
| 33 | [118](/Volumes/PortableSSD/GitHub/voiage/paper.md:118) | “Two concrete developer-led research uses” is unsupported. | Replace with “two reproducible research-readiness materials.” |
| 34 | [118–121](/Volumes/PortableSSD/GitHub/voiage/paper.md:118) | Accurate description of the synthetic health example, but it is a demonstration rather than proven research use. | Reclassify. |
| 35 | [121–122](/Volumes/PortableSSD/GitHub/voiage/paper.md:121) | Rerunnability is supported. | Retain. |
| 36 | [123–125](/Volumes/PortableSSD/GitHub/voiage/paper.md:123) | Integration-contract description is supported. | Add an accessible citation or repository link. |
| 37 | [125–126](/Volumes/PortableSSD/GitHub/voiage/paper.md:125) | Properly disclaims independent adoption. | Retain. |
| 38 | [128](/Volumes/PortableSSD/GitHub/voiage/paper.md:128) | Public development since July 2025 is verified. | Retain. |
| 39 | [128–131](/Volumes/PortableSSD/GitHub/voiage/paper.md:128) | Future-use language is aspirational and adds little screening evidence. | Remove or replace with present evidence. |
| 40 | [135–138](/Volumes/PortableSSD/GitHub/voiage/paper.md:135) | AI disclosure gives tools and scope. | Retain; add exact model/version identifiers where recoverable. |
| 41 | [138–140](/Volumes/PortableSSD/GitHub/voiage/paper.md:138) | Transparently explains historical version limitations. | Retain. |
| 42 | [140–143](/Volumes/PortableSSD/GitHub/voiage/paper.md:140) | Human validation description is appropriate. | Retain. |
| 43 | [143–145](/Volumes/PortableSSD/GitHub/voiage/paper.md:143) | Responsibility statement is appropriate. | Retain. |
| 44 | [145–146](/Volumes/PortableSSD/GitHub/voiage/paper.md:145) | Prospective promise about JOSS interactions is policy-relevant but not evidence. | Retain briefly or remove for concision. |
| 45 | [150–152](/Volumes/PortableSSD/GitHub/voiage/paper.md:150) | Acknowledgement is acceptable. | Retain. |
| 46 | [152](/Volumes/PortableSSD/GitHub/voiage/paper.md:152) | Funding declaration is complete based on author confirmation. | Retain. |
| 47 | [152–153](/Volumes/PortableSSD/GitHub/voiage/paper.md:152) | Conflict declaration is complete based on author confirmation. | Retain. |
| 48 | [157–159](/Volumes/PortableSSD/GitHub/voiage/paper.md:157) | Release and Software Heritage evidence exist, but wording should say that the tag is included in the snapshot. | Revise precisely. |
| 49 | [159–160](/Volumes/PortableSSD/GitHub/voiage/paper.md:159) | Software-citation-guidance sentence does not advance the scientific account. | Remove unless tied to a concrete metadata decision. |

## 8. Recommended replacement language

### Summary architecture paragraph

Replace [L41–48](/Volumes/PortableSSD/GitHub/voiage/paper.md:41) with:

> Version 1.0.0 is available primarily as a Python package. A shared Rust layer performs selected calculations and defines common data and diagnostic rules. The included R and Julia interfaces currently support EVPI only and require additional setup; they do not yet match the Python package’s broader feature set.

### Binding scope

Replace [L65–68](/Volumes/PortableSSD/GitHub/voiage/paper.md:65) with:

> Python is the primary installable interface. The R and Julia source packages route EVPI for net-benefit matrices to the same Rust implementation, while their broader method coverage remains narrower.

### Build-versus-contribute rationale

Replace the core of [L82–90](/Volumes/PortableSSD/GitHub/voiage/paper.md:82) with:

> `voiage` complements these tools rather than replacing them. Its present distinction is a versioned decision and result contract, together with an EVPI implementation shared across Python, R and Julia and explicit provenance and maturity metadata. These choices support contract-based interchange beyond a single language, but add packaging and interface-maintenance costs. Established R tools remain deeper for several health-economic workflows, and the `voiage` R and Julia interfaces currently expose only a subset of the Python package.

This paragraph should be accompanied by a sourced comparison with other directly relevant tools, including `dampack`, or by an explicit explanation of why such tools are outside the comparison’s defined scope.

### Cross-language consistency

Replace [L102–103](/Volumes/PortableSSD/GitHub/voiage/paper.md:102) with:

> This architecture trades a larger repository and explicit interface contracts for a testable route to matching EVPI behaviour across the present bindings.

### CI and release evidence

Replace [L110–114](/Volumes/PortableSSD/GitHub/voiage/paper.md:110) with:

> The repository exercises analytical fixtures, unit, integration, property-based, differential, mutation, fuzz, Miri, cross-language conformance, clean-install and cross-platform checks in CI. The v1.0.0 GitHub release provides a source distribution, three platform wheels and checksums; GitHub build-provenance attestations verify the released package artifacts. A CycloneDX SBOM is generated separately by the repository’s SBOM workflow.

### Research-impact evidence

Replace [L118–131](/Volumes/PortableSSD/GitHub/voiage/paper.md:118) with:

> The repository provides two reproducible materials that demonstrate research readiness rather than independent adoption. A fixed-seed synthetic health example compares parameter uncertainty and proposed study sizes under stated uptake, delay, population and cost assumptions; its script, figure and machine-readable outputs can be rerun and inspected. A versioned contract with the `vop_poc_nz` research repository exchanges decision-model records, provenance, schema fingerprints and expected numerical results between the projects [add citation or stable repository link]. Both resources were developed by the author. No independent research use is claimed.

### Availability statement

Replace [L157–160](/Volumes/PortableSSD/GitHub/voiage/paper.md:157) with:

> Version 1.0.0 is available from the public release [@voiage2026]. The corresponding release tag is included in the Software Heritage snapshot [@voiage_software_heritage].

## 9. Simulated screening disposition

**Disposition: Hold for major pre-review revision; do not invite an editor yet.**

The software appears to fall within JOSS scope and demonstrates unusually strong engineering and reproducibility infrastructure. The hold is warranted because several manuscript statements materially exceed or misdescribe the evidence, while the state-of-the-field and impact sections do not yet meet a fail-closed screening standard.

A revised manuscript could become suitable for editorial assignment after:

1. Reclassifying the two author-created materials as demonstrations or research-readiness evidence.
2. Correcting the SBOM release statement.
3. Narrowing the cross-language claims to the verified EVPI surface.
4. Strengthening the state-of-the-field and build-versus-contribute analysis.
5. Correcting bibliography metadata and adding direct sources.
6. Removing aspirational impact language.
7. Updating the manuscript date and rerunning citation, rendering and sentence-level audits.

Issue #471 should remain separately tracked. Attributable non-author installation, research use or substantive external engagement would materially reduce screening risk, but it should be supplied as genuine evidence and must not be simulated or inferred.

**Final simulated score: 780/1000.**

<oai-mem-citation>
<citation_entries>
MEMORY.md:1340-1347|note=[located prior manuscript evidence and review context]
MEMORY.md:1390-1390|note=[preserved the requested scientific framing and claim restraint]
</citation_entries>
<rollout_ids>
</rollout_ids>
</oai-mem-citation>
