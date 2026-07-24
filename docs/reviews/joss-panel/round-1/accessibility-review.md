# Applied-research accessibility review

Manuscript: [paper.md](/Volumes/PortableSSD/GitHub/voiage/paper.md)
Rubric: [rubric.md](/Volumes/PortableSSD/GitHub/voiage/docs/reviews/joss-panel/rubric.md)
Review date: 24 July 2026
Score: **688/1000**
Recommendation: **Major revision before JOSS submission**

This is an internal simulated review, not a JOSS editorial decision. No files were edited.

## Overall assessment

A non-developer can understand that `voiage` concerns decisions under uncertainty, but cannot yet understand the paper’s complete argument without software-engineering knowledge.

The manuscript does not adequately explain:

- what the four information measures mean in practical terms;
- what decision the health example represents;
- what the example found;
- why timing, uptake, population, and study cost change the research decision;
- what is distinctive about `voiage` from an applied researcher’s perspective;
- what Python, R, and Julia users can actually do;
- which claimed EVSI calculations come from the released package and which come from a separate manuscript script.

Instead, the paper devotes substantial space to Rust contracts, bindings, serialization, PyO3, C interfaces, kernels, fixtures, fuzzing, Miri, and provenance attestations. These details are meaningful to maintainers, but JOSS expressly requires the opening to address a diverse non-specialist audience and asks that software design be explained through its importance to the research application. [JOSS paper guidance](https://joss.readthedocs.io/en/latest/paper.html)

The strongest accessible evidence is absent from the paper. The synthetic health example found that:

- the programme was preferred in 49.24% of simulations at a value of 50,000 per unit of health;
- removing all uncertainty was worth about 644 per affected person;
- uncertainty about health effect mattered more than uncertainty about programme cost;
- an immediate, fully adopted study became worthwhile between 100 and 200 participants;
- a study with a two-year delay and 60% uptake became worthwhile only between 800 and 1,200 participants.

These results give policy, healthcare, business, marketing, and environmental readers an intuitive reason to care about VOI: the value of research depends on what it resolves, whom the result can affect, how quickly it is implemented, and how widely it is adopted.

## Material accessibility blockers

### 1. The paper catalogues measures without explaining them

EVPI, EVPPI, EVSI, and ENBS appear together at lines 35–38. The abbreviations are expanded, but their meanings remain opaque.

A non-specialist needs four questions:

- Could resolving all uncertainty change the decision?
- Which particular uncertainty matters most?
- How much would a proposed study reduce uncertainty?
- Would the resulting benefit exceed the study’s cost?

The formal names can follow those questions.

### 2. The paper does not contain a worked example in a reader-facing sense

Lines 118–122 say that an example exists but provide neither the decision nor any result. A reader cannot tell:

- what alternatives are being compared;
- what is uncertain;
- what “better” means;
- whether research is worthwhile;
- how the result changes with delay or uptake.

The manuscript should report one short health example, not merely point to files.

### 3. The unqualified EVSI claim is scientifically and editorially unsafe

Lines 34–39 imply that the package provides EVSI as an established capability. Repository inspection shows that the defensible normal–normal EVSI used in the health example is implemented separately in [generate_paper_health_example.py](/Volumes/PortableSSD/GitHub/voiage/scripts/generate_paper_health_example.py:76), while the script imports only package EVPI, EVPPI, and ENBS at lines 14–15.

This distinction is invisible to a non-developer. It also affects the claimed research use at lines 118–122.

Until the validated normal–normal model is part of the public API, the paper should say:

> `voiage` provides EVPI, regression-based EVPPI, ENBS, decision diagnostics, and several method-specific EVSI estimators. The health example evaluates its proposed study with a separately specified normal–normal EVSI model. Other EVSI methods require validation appropriate to their sampling models.

This is a manuscript blocker and a repository-owned scientific issue.

### 4. The distinctive value is expressed as architecture rather than research benefit

The paper’s best scholarly proposition is:

> `voiage` keeps the alternatives, units, uncertainty draws, population assumptions, study assumptions, and provenance of an analysis together when calculations or results move between tools.

That is understandable and useful. “Language-neutral decision/result contract” is not.

### 5. The impact statement overclassifies engineering evidence

The health example is a developer-created research demonstration. The `vop_poc_nz` contract is an interoperability demonstration. Calling both “research uses” at line 118 overstates the evidence.

### 6. Cross-domain relevance is asserted rather than shown

Health economics is supported by citations and an example. Policy and environmental management are named but not demonstrated. Marketing and business are not mentioned.

The paper need not provide five examples. It should explain the common decision structure:

> The approach applies when an organisation must choose among alternatives before uncertain costs, benefits, demand, health effects, environmental outcomes, or implementation conditions are fully known.

Then it should say that non-health applications remain prospective unless evidence exists.

## Argument-level revision

The manuscript should follow this argument:

1. Decisions often have to be made before important outcomes are known.
2. VOI assesses whether resolving that uncertainty could improve the decision.
3. A valid analysis depends on more than a formula: alternatives, units, uncertainty, population, timing, uptake, study design, and cost all matter.
4. Existing specialist tools provide mature workflows, particularly in R and health economics.
5. `voiage` addresses a different need by preserving these decision and result details across a broad Python workflow and selected shared calculations for R and Julia.
6. A health example demonstrates how the source of uncertainty, study size, delay, uptake, and affected population alter the value of research.
7. Python is the primary interface. R and Julia currently share only the direct EVPI calculation through Rust.
8. External adoption has not yet been demonstrated.

## Paragraph-level review

| Paragraph | Assessment | Required change |
|---|---|---|
| Summary 1, lines 30–39 | Starts accessibly but becomes an acronym catalogue | Organise around four practical questions and distinguish perfect from sample information |
| Summary 2, lines 41–48 | Dominated by implementation terminology | State the Python/R/Julia capability boundary in user terms |
| Statement of need 1, lines 52–58 | Plausible but generic | Name the information that may be lost: labels, units, groups, population, timing, uptake, and provenance |
| Statement of need 2, lines 60–68 | Mixes audience, benefits, architecture, and language support | Separate applied need from implementation choice |
| State of field 1, lines 72–80 | Health-specific comparison is understandable but incomplete | Add directly relevant tools and explain differences in ordinary language |
| State of field 2, lines 82–90 | Distinctive scope is obscured by “contracts” and “boundaries” | Explain what travels between languages and avoid speculating about other packages’ purpose |
| Software design 1, lines 94–100 | Written for developers | Replace PyO3/C/kernel detail with consequences for users and reviewers |
| Software design 2, lines 102–108 | Some useful trade-offs, but still abstract | Explain which methods are shared and why maturity labels protect applied users |
| Software design 3, lines 110–114 | Reads as a CI feature list | Summarise assurance by the errors it is intended to detect |
| Research impact 1, lines 118–126 | Misclassifies evidence and omits results | Report findings and distinguish research from interoperability demonstrations |
| Research impact 2, lines 128–131 | Mostly prospective | State the demonstrated boundary and unresolved external-use evidence |
| AI disclosure, lines 135–146 | Complete but long and procedural | Tighten while preserving tools, uses, verification, responsibility, and communication boundary |
| Acknowledgements and References, lines 148–160 | Generic acknowledgement; availability is under References | Retain declarations and add a separate Software and data availability section |

## Sentence-level inventory

### Metadata

- **Line 2:** The title is understandable, although “Implementation” is broad. A clearer alternative is:
  > `voiage: Value of Information for Research and Decision Making`

- **Lines 4–9:** The tags foreground implementation languages but omit research prioritisation and decision support. Consider replacing `Python` and `Rust` with `research prioritisation` and `decision support`, unless language tags are important for JOSS discovery.

### Summary

1. **Lines 30–31 — revise.** The sentence conflates perfect-information and sample-information analysis.

   > Value of Information (VOI) analysis estimates the expected benefit of resolving uncertainty before choosing among alternatives.

2. **Lines 31–34 — revise.** “Must” is unnecessarily strong, and the domain list is unsupported within the paper.

   > It is useful when decisions are made before their costs, benefits, effects, demand, or implementation conditions are fully known, including in healthcare, policy, business, marketing, and environmental management.

3. **Lines 34–39 — replace.** This is an inaccessible feature catalogue and overstates EVSI.

   > `voiage` helps analysts ask whether uncertainty could change a decision, which uncertain quantities matter most, and whether the expected benefit of a proposed study exceeds its cost. It provides perfect-information and partial-perfect-information measures, study-value calculations under specified models, expected net benefit of sampling, and decision diagnostics.

4. **Lines 41–43 — replace.** “Binding-independent,” “domain contracts,” “serialization,” and “kernels” are unexplained engineering terms.

   > Python is the main interface in version 1.0.0. Shared Rust components validate selected inputs and perform calculations intended to remain consistent across supported languages.

5. **Lines 43–46 — revise.**

   > Python provides the broadest set of modelling, checking, reporting, and plotting tools. The current R and Julia packages directly share only the EVPI calculation and require additional source or native-library setup.

6. **Lines 46–48 — replace.**

   > The documentation states these differences so that users know which results come from the same calculation and which features are available only in Python.

### Statement of need

7. **Lines 52–56 — replace.** “Fragmented” does not tell readers what the practical problem is.

   > Existing VOI workflows span specialist packages, programming languages, web tools, and model-output formats. Moving an analysis between them requires analysts to preserve alternatives, labels, units, uncertainty draws, parameter groups, population assumptions, and study assumptions.

8. **Lines 56–58 — revise.**

   > If those details are lost or interpreted differently, two tools can attach different meanings to what appears to be the same result.

9. **Lines 60–62 — replace.** “One clear description” and “careful data checks” are vague self-evaluations.

   > `voiage` records the alternatives, uncertain inputs, population and study assumptions, warnings, and results in reviewable forms, and rejects malformed or inconsistent inputs before calculation.

10. **Lines 62–65 — retain with minor simplification.**

   > It is intended for researchers and analysts deciding whether proposed data collection is worthwhile, comparing choices under uncertainty, or incorporating VOI into a wider evidence assessment.

11. **Lines 65–68 — replace.**

   > Python currently provides the complete workflow. R and Julia users can call the same Rust implementation for EVPI, but their direct interfaces do not yet match Python’s wider functionality.

### State of the field

12. **Lines 72–73 — retain, but broaden the explanation.**

   > VOI is well established in decision analysis and health economics, where it is used to assess the consequences of uncertainty and the potential value of further research [@claxton1999irrelevance; @ades2004evsi].

13. **Lines 74–78 — expand and correct.**

   > Existing R tools provide mature health-economic workflows. `voi` supports several approaches to EVPI, EVPPI, EVSI, and ENBS [@voi_cran2024]; `BCEA` combines cost-effectiveness analysis with graphical reporting [@green2022bcea]; and `dampack` supports decision-model analysis and VOI. SAVI provides a web-based workflow for regression-based EVPPI, but the current citation supports the regression method rather than the SAVI software itself.

14. **Lines 78–80 — retain.**

   > These tools remain appropriate when their methods, language, and reporting conventions fit the research question.

15. **Line 82 — retain.**

   > `voiage` does not seek to replace these tools.

16. **Lines 82–85 — replace.**

   > It was developed for a different requirement: keeping the description of a decision, its assumptions, its provenance, and selected calculations consistent across Python, R, and Julia.

17. **Lines 86–88 — replace.** The current sentence speculates about other maintainers’ intentions.

   > Extending an R-centred workflow would not by itself provide the required language-neutral interface, while separate implementations in each language would require independent checks for numerical agreement.

18. **Lines 88–90 — revise.**

   > This approach trades some of the methodological depth of specialist tools for portable inputs and results. Python currently provides much more functionality than the R and Julia interfaces.

### Software design

19. **Line 94 — replace.**

   > The design separates calculations intended to be shared across languages from modelling, plotting, and reporting that remain language-specific.

20. **Lines 95–97 — replace.**

   > Rust contains the shared data rules and selected calculations. This allows the same EVPI calculation to be tested independently and called from more than one language.

21. **Lines 97–100 — replace.**

   > Python manages labelled data and provides the wider set of methods. The current R and Julia packages call the shared Rust EVPI calculation; they do not yet provide equivalent access to the full Python workflow.

22. **Lines 102–105 — replace.**

   > Maintaining shared calculations and multiple language interfaces increases release and compatibility work. In return, reviewers can test the calculation independently and compare results across supported languages.

23. **Lines 105–106 — either remove or simplify.**

   > Plotting and other optional features are kept out of the basic installation.

24. **Lines 106–108 — replace.**

   > Methods are labelled as stable, developing, or experimental to show the strength of their validation. A working implementation does not establish that a method is appropriate for every research question.

25. **Lines 110–113 — replace.** “Correctness evidence” is too strong, and the list is inaccessible.

   > Software-assurance checks cover known analytical results, invalid inputs, consistency between implementations, repeatability, installation, and supported operating systems. Additional stress tests examine unusual inputs and memory or concurrency errors.

26. **Lines 113–114 — correct.** The public GitHub release contains source, wheels, and checksums, but no SBOM asset was listed.

   > The version 1.0.0 GitHub release contains source archives, platform wheels, and checksums. Separate workflows generate provenance attestations and retain software-bill-of-materials evidence.

### Research impact statement

27. **Line 118 — replace.**

   > The repository contains one synthetic research demonstration and one interoperability demonstration.

28. **Lines 118–121 — replace.** The existing sentence incorrectly implies that package EVSI produced the study-size comparison.

   > The synthetic health example compares a programme with current practice when both its health effect and cost are uncertain. It uses `voiage` for EVPI, EVPPI, and ENBS, together with a separately specified normal–normal EVSI model for the proposed study.

29. **Lines 121–122 — replace with findings.**

   > At a value of 50,000 per unit of health, the programme was preferred in 49.24% of simulations. Uncertainty about health effect had greater partial value than uncertainty about programme cost, and delayed or incomplete uptake substantially increased the study size required for research to become worthwhile.

30. **Lines 123–125 — revise.**

   > A separate contract with the `vop_poc_nz` repository demonstrates how decision records, provenance, schema identifiers, and expected numerical results can be transferred between projects.

31. **Lines 125–126 — retain with sharper classification.**

   > These are reproducible, developer-created materials. They are not evidence of independent adoption.

32. **Line 128 — retain.**

   > The package has been developed publicly since July 2025.

33. **Lines 128–131 — replace.**

   > The same decision structure may apply in healthcare, policy, environmental management, marketing, and business when uncertain evidence can be linked to explicit alternatives and outcomes. The project has not yet documented attributable non-author research use, and non-health applications remain prospective.

### AI usage disclosure

34. **Lines 135–138 — simplify.**

   > OpenAI Codex and Google Jules assisted with repository analysis, code and test drafting, refactoring, documentation, workflow review, and manuscript editing.

35. **Lines 138–140 — retain in shortened form.**

   > Exact model versions were not retained for all historical sessions.

36. **Lines 140–143 — revise.**

   > The author selected the research problem and architecture, reviewed merged changes, and checked AI-assisted output against the source code, tests, references, numerical examples, and generated artefacts.

37. **Lines 143–146 — retain.**

   > The author is responsible for the software, claims, citations, authorship, and submission. No AI system is an author, and AI tools will not compose substantive exchanges with JOSS editors or reviewers.

### Acknowledgements and availability

38. **Lines 150–152 — remove.** The generic acknowledgement does not identify a person, organisation, or contribution.

39. **Line 152 — replace.**

   > This work received no external funding, and no sponsor influenced its design or reporting.

40. **Lines 152–153 — retain.**

   > The author declares no competing interests.

41. **Lines 157–160 — move and replace under `# Software and data availability`.**

   > `voiage` version 1.0.0, the fixed-seed health-example script, and its synthetic outputs are available from the public repository and release [@voiage2026]. The reviewed source is preserved by Software Heritage [@voiage_software_heritage].

The software-citation guidance reference can be removed unless the manuscript discusses the citation metadata as part of its scholarly contribution.

## Rubric score

| Dimension | Score | Deductions |
|---|---:|---|
| Scope, significance, and research use | **132/180** | −20 for classifying interoperability as research use; −15 for mostly prospective cross-domain significance; −13 because the applied contribution is obscured by architecture |
| Statement of need and audience | **87/120** | −15 for an abstract fragmentation problem; −10 for failing to define the information that must be preserved; −8 for limited accessibility to non-health audiences |
| State of the field and build-versus-contribute case | **83/130** | −15 for incomplete comparison; −10 for health-only framing; −12 for unsupported claims about other packages’ purpose; −10 for explaining differences through contracts and boundaries rather than user needs |
| Scientific and numerical accuracy | **103/150** | −25 for the unqualified EVSI capability claim; −10 for conflating perfect and sample information; −7 for inaccurately describing the health example’s package use; −5 for omitting the verified findings |
| Software design and research relevance | **54/100** | −28 for unexplained engineering terminology; −10 for insufficient connection to research practice; −8 for not clearly delimiting language capabilities |
| Reproducibility, packaging, documentation, and tests | **84/100** | −7 for inaccurate release/SBOM wording; −5 for inaccessible assurance terminology; −4 because reviewer-facing installation implications are not summarised |
| Research-impact statement | **38/80** | −20 for overclassifying evidence; −12 for reporting no result; −6 for relying on prospective applications; −4 because the EVSI path is incorrectly attributed |
| Structure, metadata, and JOSS format | **56/60** | −2 for availability material under References; −2 because the example is nominal rather than substantive |
| Clarity, accessibility, and sentence quality | **31/55** | −12 for unexplained jargon; −6 for acronym density; −4 for catalogue-like construction; −2 for vague self-evaluations |
| Citations, provenance, declarations, and AI disclosure | **20/25** | −2 for indirect SAVI support; −2 for unsupported cross-domain/fragmentation claims; −1 for unnecessary software-citation guidance |
| **Total** | **688/1000** | **Major revision** |

The fail-closed blocker cap applies because the manuscript contains a material mismatch between its EVSI claims and the implementation used by the worked example.

## Manuscript defects versus other gates

### Manuscript defects

- The summary is not accessible enough for JOSS’s non-specialist requirement.
- VOI measures are named but not explained.
- Perfect and sample information are conflated.
- The EVSI capability claim is overbroad.
- The worked health example reports no results.
- The package’s distinctive research value is expressed through engineering abstractions.
- Interoperability evidence is misclassified as research use.
- Cross-domain applicability is not adequately bounded.
- The field comparison and build-versus-contribute case are incomplete.
- The GitHub release/SBOM statement is inaccurate.
- Availability material is placed under References.

### Repository-owned defects

- The validated normal–normal EVSI example is not exposed through the public API.
- Public EVSI estimators need analytical-reference, simulation-recovery, PSA-size-invariance, and likelihood-sensitivity validation before generic stable-EVSI claims are defensible.
- R and Julia installation remains dependent on source/native-library arrangements, and broader method parity is absent.

### External gates

- [Issue #471](https://github.com/edithatogo/voiage/issues/471) remains open without attributable non-author installation or research-use evidence.
- A permanent arXiv identifier remains externally pending, although JOSS does not require one before submission.
- JOSS scope screening, peer review, and acceptance remain editorial decisions.
- The review-version DOI archive is an acceptance-stage requirement.

The next manuscript should be organised around the decision problem and worked result. Engineering details should remain only where they explain an observable benefit or limitation for applied researchers.

<oai-mem-citation>
<citation_entries>
MEMORY.md:1390-1390|note=[Used established preference for domain-first manuscript framing and restrained claims]
MEMORY.md:1340-1351|note=[Used prior manuscript provenance to distinguish current JOSS review from earlier arXiv work]
</citation_entries>
<rollout_ids>
019f5676-c3ee-7fe1-a49b-0476d3dba926
</rollout_ids>
</oai-mem-citation>
