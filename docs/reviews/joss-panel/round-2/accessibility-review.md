# Applied-research accessibility review: round 2

Role: applied-research reader and reviewer across policy, healthcare,
marketing, business, economics, and data science

Reviewed state: branch `codex/joss-panel-review`, commit
`1dfcd3d42af591ca15fcc03ad958123cc153dbbf`, plus the uncommitted worktree
present on 24 July 2026

Score: **824/1000**

Recommendation: **major revision before JOSS submission**

Fail-closed status: **manuscript, citation, release-identity, and
reviewer-documentation blockers remain; score capped below 950**

This is an internal simulated review, not a JOSS editorial decision or an
acceptance prediction.

## Accessibility verdict

Round 2 is a material improvement. A reader without software-development
knowledge can now understand the four questions posed by Value of Information
(VOI), the broad purpose of `voiage`, the difference between its Python and
other language interfaces, and the direction of the worked health-example
results. The writing is restrained and does not advertise the software as
methodologically superior.

The manuscript is not yet self-contained enough for an applied reader to
interpret or reproduce its central example. It says that expected net benefit
of sampling (ENBS) becomes positive under "the stated population and cost
assumptions", but the paper never states those assumptions. They are recoverable
only from the Python generator. The paper also uses quality-adjusted life year
(QALY), net benefit, prior, likelihood, posterior update, numerical parity,
provenance, and normal--normal without explaining them in ordinary language.
These are familiar to some health economists and data scientists, but not to
the full audience named in this review.

The result is uneven by audience:

| Reader | Current accessibility | Reason |
| --- | --- | --- |
| Healthcare and health economics | Partial to good | The decision and outcomes are recognisable, but QALY is not expanded, the study informs only health effect without saying so, and the population and study-cost assumptions are absent. |
| Policy and research funding | Partial | The four practical questions are clear, but the paper does not explain how affected population, timing, uptake, and discounting convert per-person study value into a funding decision. |
| Economics | Partial | The uncertainty framework is intelligible, but net benefit and "value units" are undefined and the reported estimates are not identified as simulation estimates or analytical values consistently. |
| Marketing and business | Weak to partial | The text says that non-health outcomes are allowed, but gives no plain example involving demand, customer response, revenue, cost, or implementation. A second worked example is unnecessary; one bounded sentence would be enough. |
| Data science | Partial to good | The probabilistic model and validation rationale are intelligible, but the exact scientific scope of the EVSI interfaces is difficult to recover because the paper, current documentation, current worktree, and release do not yet describe the same interface. |

The body-only readability diagnostic returned Flesch Reading Ease 21.8,
Flesch-Kincaid grade 14.3, and SMOG 15.3. These formulae are not scientific
quality thresholds and are inflated by unavoidable method names. They do,
however, support the sentence-level finding that the software-design and study
model paragraphs require more translation.

## Evidence independently checked

- the complete current `paper.md` and `paper.bib`;
- all three machine-readable health-example outputs and the current figure;
- `scripts/generate_paper_health_example.py`;
- the current public Python EVSI and EVPPI implementations, Rust
  normal--normal EVSI kernel, tests, and method documentation;
- the public signed `v1.0.0` tag and GitHub release assets;
- the current JOSS manuscript validator and focused scientific-contract tests;
- the Software Heritage identifier and release-readiness material;
- current official JOSS paper and review guidance;
- the official SAVI interface and the published review of web-based VOI tools;
  and
- DOI metadata for the cited ISPOR task-force paper.

The repository-owned JOSS validator passed. Thirteen focused JOSS-readiness and
EVSI scientific-contract tests passed. Recalculation from the current generator
reproduced every value quoted in the paper:

| Quantity | Recalculated value |
| --- | ---: |
| Programme preferred | 0.4924 |
| EVPI per person | 644.153547 |
| Health-effect EVPPI per person | 589.666167 |
| Programme-cost EVPPI per person | 249.594994 |
| EVSI per person at total sample size 200 | 124.179366 |
| Immediate/full-uptake ENBS at 100 and 200 | -225,618.06; 157,057.73 |
| Two-year-delay/60%-uptake ENBS at 800 and 1,200 | -73,757.03; 104,031.41 |

The numbers are reproducible in the current worktree. Their presentation, model
boundary, and release identity are the defects.

## Rubric score and deductions

| Dimension | Score | Deduction |
| --- | ---: | --- |
| Scope, significance, and research use | 152/180 | The use case is plausible and bounded, but the paper documents only a synthetic example and a same-author interoperability exercise. It does not yet show a completed applied analysis or independent use, and non-health relevance is asserted rather than demonstrated. |
| Statement of need and audience | 107/120 | The transfer problem and intended users are clear. "Probabilistic decision model", "provenance", "inspectable objects", and "value units" still require translation, and policy/business/marketing readers receive no concrete bridge from their decisions to the VOI structure. |
| State of the field and build-versus-contribute case | 110/130 | The comparison is fair and now includes R, web, and Python options. The rationale becomes engineering-heavy at "language-neutral interface" and "numerical-parity maintenance", and it omits the directly relevant review of web-based VOI tools and the broader tool categories in that review. |
| Scientific and numerical accuracy | 119/150 | The calculations reproduce, but the paper omits material assumptions behind EVSI and ENBS, does not define QALY or net benefit, reports Monte Carlo estimates with unjustified precision, and does not identify EVPPI as regression-based. |
| Software design and research relevance | 83/100 | The high-level split is understandable. The maturity paragraph moves abruptly into prior, likelihood, outcome variance, backend, and posterior terminology without explaining the practical protection these boundaries give an analyst. |
| Reproducibility, packaging, documentation, and tests | 80/100 | Local focused checks pass and the data are present, but `v1.0.0` does not contain the featured public normal--normal EVSI path, the exact reviewed revision is still future tense, and the current EVSI documentation contradicts the callback-based generic interface. |
| Research-impact statement | 61/80 | The example reports useful results and the non-adoption boundary is honest. Both uses are author-created, the first is synthetic, and the second demonstrates transfer machinery rather than a substantive research finding. |
| Structure, metadata, and JOSS format | 56/60 | Required sections and metadata are present and the validator reports a compliant body length. Limitations are dispersed rather than gathered into a short reader-facing statement, and the availability section contains workflow language rather than final evidence. |
| Clarity, accessibility, and sentence quality | 41/55 | The opening is much clearer, but several dense lists and undefined domain terms remain. The figure is visually legible at source resolution, although its QALY label and ENBS assumptions are not explained in the text or caption. |
| Citations, provenance, declarations, and AI disclosure | 15/25 | Funding, conflicts, affiliations, archive, and AI disclosure are present. The Rothery record misnames two authors, the web-tool comparison lacks the directly relevant review, `vop_poc_nz` is cited only through a mutable repository root, and no immutable reviewed software revision is cited. |
| **Total** | **824/1000** | **Major revision; fail-closed conditions apply.** |

## Fail-closed blockers

### B1. The worked example is not scientifically interpretable from the paper

Lines 123--138 provide a prior for health effect and a study sample size, then
report population ENBS under "the stated population and cost assumptions".
Those assumptions are not stated. Source inspection establishes that the
calculation assumes:

- independent Normal uncertainty for incremental health effect
  \(N(0.060, 0.030^2)\) and programme cost \(N(3000, 650^2)\);
- incremental net benefit of \(50{,}000\theta-C\);
- an equal-allocation two-arm study with known individual outcome standard
  deviation 1.0;
- a study that informs health effect but not programme cost;
- 1,300 eligible people per year for ten years;
- 3% annual discounting;
- study cost of 1,200,000 plus 100 per participant; and
- a delayed scenario in which evidence affects years 3--10 and reaches 60% of
  the eligible population.

Without these values, an applied reader cannot tell why research becomes
worthwhile, transfer the example to a new setting, or distinguish a change in
information from a change in implementation. This is a manuscript blocker.

Replace the first two research-impact paragraphs with a compact assumptions
table or prose with this minimum content:

> The synthetic example compares a new programme with current practice.
> Current uncertainty about incremental health gain is Normal with mean 0.06
> quality-adjusted life years (QALYs) and standard deviation 0.03. Incremental
> programme cost is independently Normal with mean 3,000 value units and
> standard deviation 650. At 50,000 value units per QALY, incremental net
> benefit is the value of the health gain minus programme cost.
>
> The proposed two-arm study informs health gain, not programme cost. It
> allocates participants equally and assumes a known individual outcome
> standard deviation of 1 QALY. Population ENBS assumes 1,300 eligible people
> per year for ten years, 3% annual discounting, and study cost of 1,200,000
> plus 100 per participant. The delayed scenario assumes that usable evidence
> begins in year 3 and reaches 60% of eligible people.

Use "value units" only after saying that they stand in for a currency or another
explicit decision-maker value scale in this synthetic example.

### B2. The only cited release does not contain the featured EVSI implementation

Lines 42--43 and 107--110 describe a public Rust-owned normal--normal EVSI
calculation in the current worktree. Tag `v1.0.0`, commit
`05cc373d78ae74143194e889ff1317de4dfea52e`, predates
`normal_normal_two_arm_evsi`. Lines 173--178 nevertheless identify `v1.0.0` as
the public software and promise that the reviewed revision "will be frozen".

An applied reviewer following the citation cannot install the software
described by the paper. Commit, test, release, and cite the exact revision, or
limit the manuscript to what `v1.0.0` actually provides. The final sentence
needs factual identifiers, for example:

> Version X.Y.Z at commit `<full commit>` is the version reviewed for this
> paper. Its release contains the fixed-seed script and synthetic outputs; the
> release-evidence manifest records their checksums.

Do not insert placeholders into the submission. Insert the actual identifiers
only after the release and hosted evidence exist.

### B3. Reviewer-facing EVSI documentation contradicts the current interface

The current public function requires explicit `trial_simulator` and
`posterior_sampler` callbacks for `method="two_loop"` so that the analyst
declares the sampling model and preserves joint dependence. The current
`docs/astro-site/src/content/docs/methods/evsi.mdx` instead says that the
"default" path itself recognises `mean_<arm>` and `sd_outcome` and performs the
update. That describes the retired implicit path, not the current interface.

This is a repository defect rather than a false numerical result in the paper,
but it blocks an applied reviewer from understanding the method boundary. Align
the method page, API reference, examples, and manuscript before submission.
Explain in plain language that:

1. the named normal--normal function is a narrow analytical model with explicit
   assumptions;
2. generic two-loop EVSI requires user-supplied study simulation and posterior
   updating;
3. those callbacks carry correlated parameters jointly; and
4. compatibility estimators do not expose a complete validated study-model
   contract.

### B4. A foundational citation has incorrect author names

`paper.bib` line 24 gives "John F. Murray" and "Gary D. Sanders Schmidler".
DOI metadata for `10.1016/j.jval.2020.01.004` gives **James F. Murray** and
**Gillian D. Sanders Schmidler**. Correct the record. Under the panel rubric, an
unresolved citation defect is fail-closed.

### B5. The availability statement is an unfinished promise

Lines 177--178 say that the exact revision and evidence "will be frozen before
submission". This tells a reader what the author intends to do, not what was
reviewed. Replace it with the exact tag, commit, release-evidence manifest, and
archive identifier after they exist. If Software Heritage does not yet contain
the final release, state which revision the current SWHID resolves and request a
new snapshot rather than implying that it archives the reviewed version.

## Whole-document and paragraph audit

| Lines | Assessment | Required change |
| ---: | --- | --- |
| 2--25 | Mostly clear | Retain the title. Add discovery terms such as `research prioritisation` and `decision support`; consider whether language tags are more useful than audience tags. Confirm that every affiliation is current at submission. |
| 30--37 | Accessible | The four-question structure works. Keep it. The expansion of four acronyms in one sentence is dense but justified because the questions have already supplied their meanings. |
| 39--45 | Partly accessible; release-blocked | Replace "probabilistic decision models", "provenance", and "shared data rules" with plainer descriptions. Retain the honest Python/R/Julia capability boundary. Bind the EVSI sentence to a release that contains it. |
| 49--55 | Strong | The transfer problem is concrete. The long assumption list is acceptable because every item changes interpretation, although it could be split once for rhythm. |
| 57--63 | Needs translation | "Inspectable objects" and "malformed or inconsistent inputs" do not tell an applied reader what is inspected or rejected. Give two concrete validation examples and one non-health decision example. |
| 67--81 | Mostly clear | The tool descriptions are concise and fair. Add Tuffaha et al.'s review of web-based VOI tools and explain whether BCEAweb, RANE, or VICTOR are relevant categories rather than silently narrowing the comparison to SAVI. |
| 83--89 | Needs translation | "Provenance", "language-neutral interface", and "numerical-parity maintenance" express the package rationale in maintainer language. Explain instead what a researcher can transfer and why one shared calculation reduces disagreement between language implementations. |
| 93--100 | Good with one simplification | The division of responsibilities is intelligible. Replace "same selected calculation" and "parity tests" with "the same EVPI calculation" and "checks that each language returns the same result". |
| 102--110 | Scientifically responsible but inaccessible | Keep the maturity boundary, but translate prior, likelihood, posterior update, and backend into the assumptions an analyst supplies and the risks the checks prevent. Correct the blanket "without a declared likelihood" description of compatibility estimators. |
| 112--119 | Too catalogue-like | Retain one sentence explaining the failures the test suite is designed to catch. Keep release assets and software-component inventory only after they refer to the reviewed release. |
| 123--138 | **Blocker** | State all decision, prior, study, population, timing, uptake, discount, and cost assumptions; define QALY and net benefit; identify simulated versus analytical results; report uncertainty for simulation estimates; use "illustrate" rather than "show". |
| 140--143 | Figure understandable but under-captioned | Explain what the vertical reference, dashed EVPI line, and zero ENBS line mean. State that the plotted values are estimates from 10,000 synthetic draws except for analytical EVSI. |
| 145--151 | Honest but technical | Replace "compatibility contract", "schema identifiers", and "provenance" with a plain account of what files and expected results move between projects. Cite an immutable release, tag, or commit rather than the mutable repository root. |
| 153--164 | Clear and complete | Retain. The disclosure is necessarily procedural, but it states tools, uses, verification, human decisions, responsibility, and authorship directly. |
| 166--169 | Clear | Retain, subject to final author confirmation. |
| 171--178 | **Blocker** | Cite the exact reviewed software and data revision, identify where the synthetic outputs live, and replace the future-tense freeze statement with completed evidence. |

## Complete substantive-sentence inventory

Every substantive sentence and the figure-caption fragment in the current
manuscript was audited below.

| # | Lines | Finding | Exact action |
| ---: | ---: | --- | --- |
| 1 | 2 | Clear title. | Retain. |
| 2 | 30--31 | Accurate high-level definition, although "expected benefit" remains abstract until the questions that follow. | Retain with the next sentence. |
| 3 | 31--34 | The four questions are the strongest non-specialist explanation in the paper. | Retain. |
| 4 | 35--37 | Correct mapping, but acronym-dense. | Retain because each acronym has just been defined by its practical question; do not add more terminology here. |
| 5 | 39--41 | "Probabilistic decision models", "uncertainty draws", and "provenance" are not plain language. | Replace the opening with: "`voiage` calculates these measures from simulated decision-model results and keeps the alternatives, units, assumptions, warnings, and record of where the inputs came from with each result." |
| 6 | 41--42 | Clear interface priority. | Retain. |
| 7 | 42--43 | Understandable after translation, but false for the cited release. | After release, use: "Rust supplies the shared EVPI calculation and a narrowly specified EVSI calculation for a two-arm study with normally distributed outcomes." |
| 8 | 43--44 | Clear and appropriately limited. | Retain. |
| 9 | 44--45 | "Shared and language-specific calculations" is accurate but abstract. | Prefer: "Users can therefore see which calculations are identical across languages and which are currently available only through Python." |
| 10 | 49--51 | Clear applied problem. | Retain. |
| 11 | 51--52 | Effective short sentence. | Retain. |
| 12 | 52--54 | Material assumptions are named, but the list is long. | Retain the content; split after "parameter groups" if space permits. |
| 13 | 54--55 | Clear consequence. | Retain. |
| 14 | 57--58 | "Inspectable objects" and "malformed or inconsistent" are vague technical self-descriptions. | Replace with a specific bounded claim, such as: "`voiage` stores labels, units, assumptions, warnings, and results together, and checks that strategy names, sample dimensions, units, and population inputs agree before calculation." Verify each named check against the implemented schema before using this text. |
| 15 | 58--60 | Audience and tasks are clear. | Retain. |
| 16 | 60--62 | Cross-domain flexibility is plausible but "outcome and value units supplied" is abstract. | Add: "For example, an outcome could be health gain, demand, revenue, emissions, or another quantity that the decision maker can place on a common value scale." Label non-health uses as prospective unless documented. |
| 17 | 62--63 | Clear boundary. | Retain. |
| 18 | 67--68 | Accurate field context. | Retain. |
| 19 | 69--70 | Clear comparator statement. | Retain. |
| 20 | 70--72 | Clear comparator statement. | Retain. |
| 21 | 72--74 | Clear comparator statement supported by the SAVI site. | Retain, and cite the Tuffaha et al. web-tool review nearby. |
| 22 | 74--76 | Fair and restrained. | Retain. |
| 23 | 78 | Useful transition but generic. | Retain with the next sentence. |
| 24 | 78--81 | The distinction between the Python packages is intelligible. | Retain; clarify that `trd-cea-toolkit` is by the same author if it is being used as evidence about the field rather than merely catalogued. |
| 25 | 83--85 | The package's distinguishing purpose is clear, but "provenance" and "language boundaries" are developer terms. | Replace with: "`voiage` instead keeps the decision description, assumptions, source record, and selected calculations together when an analysis moves between Python, R, and Julia." |
| 26 | 85--87 | "Language-neutral interface" and "numerical-parity maintenance" obstruct the reason. | Replace with: "Adding another R workflow would not make the same calculation directly available in the other languages, while separate implementations could return different results unless each were checked independently." |
| 27 | 87--89 | Important limitation stated plainly. | Retain. |
| 28 | 93--94 | Clear design boundary. | Retain. |
| 29 | 94--95 | "Common types" is vague. | Prefer: "Rust defines the shared input rules and performs selected calculations." |
| 30 | 95--96 | Clear. | Retain. |
| 31 | 97--98 | Clear. | Retain. |
| 32 | 98--100 | The trade-off is sound but "same selected calculation" and "parity tests" are indirect. | Replace with: "This avoids maintaining separate EVPI implementations, but each language package still needs installation and agreement checks." |
| 33 | 102--104 | Responsible maturity rule, expressed at an appropriate level. | Retain after replacing "independent numerical reference" with "a result calculated independently of the implementation" if space permits. |
| 34 | 104--106 | "Optional computing backend" is a developer term. | Replace with: "The documentation also says when a result is approximate or requires optional software." |
| 35 | 106--107 | Important caution. | Retain. |
| 36 | 107--110 | Scientifically specific but inaccessible and not enough to reconstruct the study. | Replace with: "The analytical EVSI method applies to an equally allocated two-arm study, normally distributed current uncertainty and study outcomes, known outcome variability, and a linear relationship between the outcome and net benefit. These assumptions determine how the study changes uncertainty." |
| 37 | 109--110 | The blanket likelihood claim is not accurate for every compatibility path. | Replace with: "Other EVSI estimators remain available for older workflows, but they do not expose a complete, validated study-model contract and are not labelled stable." |
| 38 | 112--114 | Clear purpose but list-heavy. | Retain or shorten to: "Tests compare results with known calculations, reject invalid data, check repeatability across implementations, and exercise clean installations on supported operating systems." |
| 39 | 114--116 | Mostly a continuous-integration catalogue. | Remove unless the listed test classes support a specific research risk not covered by sentence 38. |
| 40 | 116--117 | Factually correct for the public release assets. | Retain only after changing the version to the reviewed release. |
| 41 | 117--119 | "Build-provenance attestations" is unexplained. | Prefer: "GitHub records how release files were built, and a separate workflow lists their software components." |
| 42 | 123--124 | "Synthetic research demonstration" is formal and overclassifies an example. | Replace with: "The repository contains a synthetic worked example and a same-author research-workflow integration." |
| 43 | 124--126 | Clear decision, although "health effect" needs a unit. | Retain after identifying the health outcome as QALYs. |
| 44 | 126--128 | "Normal prior" is undefined and the cost distribution is omitted. | Use the first assumptions paragraph under B1, including the independent cost distribution. |
| 45 | 127--128 | Study assumptions remain incomplete and unitless. | Use the second assumptions paragraph under B1 and state that the study informs health gain but not programme cost. |
| 46 | 130--131 | Correct fixed-seed estimate but overprecise and does not show simulation uncertainty. | Report "49.2% (bootstrap 95% interval 48.2% to 50.2%)" and call it a fixed-seed simulation estimate. |
| 47 | 131--132 | Correct estimates, but EVPPI's regression basis and simulation uncertainty are hidden. | Use: "Estimated EVPI was 644 value units per person (bootstrap interval 624 to 658); regression-based EVPPI was 590 for health effect (569 to 603) and 250 for programme cost (229 to 265)." |
| 48 | 132--133 | Correct analytical result once the study and sample size are clear. | Retain as: "For a total sample of 200, the analytical study model gives EVSI of 124 per person." |
| 49 | 133--135 | Numerically correct but refers to assumptions absent from the paper. | State the B1 assumptions first, then retain the crossover statement. |
| 50 | 135--136 | Numerically correct but "two-year delay" needs the year-3 implementation interpretation. | Use: "If usable evidence begins in year 3 and reaches 60% of eligible people, ENBS changes sign between 800 and 1,200 participants." |
| 51 | 136--138 | "Show" implies general evidence from one synthetic construction. | Replace "show" with "illustrate" and end with: "The example illustrates why a research decision depends on what the study learns, who can benefit, when evidence becomes usable, how widely it is adopted, and what the study costs." |
| 52 | 140 | Caption title fragment is acceptable. | Retain. |
| 53 | 140--143 | Caption describes topics but not how to read the reference lines. | Add: "In panel A, the vertical line marks 50,000 value units per QALY; in panel B, the dashed line is the value of resolving all uncertainty; in panel C, values above zero indicate that expected population benefit exceeds study cost." |
| 54 | 142--143 | Synthetic-data boundary is clear. | Retain, while adding that EVPI and EVPPI use 10,000 fixed-seed draws and EVSI is analytical. |
| 55 | 145--148 | "Compatibility contract", "provenance", and "schema identifiers" are inaccessible. | Replace with: "In the author's `vop_poc_nz` health-economic project, a versioned data and expected-results bundle transfers the decision description and calculation checks between repositories." Cite an immutable revision. |
| 56 | 148--149 | Honest distinction between developer and independent use. | Retain. |
| 57 | 150--151 | Honest and concrete external-use boundary. | Retain. |
| 58 | 155 | Clear disclosure opening. | Retain. |
| 59 | 155--158 | Detailed but required by current JOSS policy. | Retain. |
| 60 | 158--159 | Transparent limitation. | Retain. |
| 61 | 159--162 | Human decisions and verification are clearly stated. | Retain. |
| 62 | 162--164 | Responsibility and non-authorship are direct. | Retain. |
| 63 | 168 | Clear funding declaration. | Retain after final author confirmation. |
| 64 | 168--169 | Clear conflict declaration. | Retain after final author confirmation. |
| 65 | 173 | Public-release fact is correct, but this is not the version containing the software described. | Replace with the final reviewed release and cite both the package registry and immutable release or archive as appropriate. |
| 66 | 173--174 | Clear data-status statement. | Retain and add repository-relative paths or one citation that resolves to the exact script and outputs in the reviewed revision. |
| 67 | 175--177 | Concrete SWHID, but its relation to the eventual reviewed release is unclear. | Retain only with a statement of the revision the SWHID captures, or replace it with the final release snapshot. |
| 68 | 177--178 | Future workflow language, not availability evidence. | Replace with completed tag, commit, manifest, and archive identifiers as specified in B2 and B5. |

## Figure and data accessibility

The source PNG is legible, uses both colour and line/marker differences, and
has descriptive panel titles. The panel sequence is useful:

1. whether the preferred programme changes with the value placed on health;
2. which uncertainty matters more; and
3. whether study value exceeds study cost under two implementation scenarios.

Four changes are needed:

- spell out quality-adjusted life year before the first use of QALY;
- explain all reference lines in the caption;
- state the assumptions that generate panel C in the paper or a compact table;
- identify panel A and B values as finite-simulation estimates and avoid
  displaying unsupported precision in prose.

The data files support uncertainty intervals for the simulation estimates, but
the paper does not use them. Reporting rounded intervals would make the
near-50% decision uncertainty easier to interpret and prevent readers from
treating 49.24% as an exact population quantity.

The figure's accessible text should describe the result rather than only list
the panels. Suggested alt text:

> In a synthetic comparison of a new programme with standard care, the
> programme is preferred in about half of simulations at 50,000 value units per
> QALY. Uncertainty about health effect has greater value than uncertainty about
> programme cost. A study becomes worthwhile between 100 and 200 participants
> with immediate full uptake, but only between 800 and 1,200 participants when
> evidence is delayed for two years and uptake is 60%.

## Citation and comparison corrections

1. Correct the two author names in `rothery2020voi`.
2. Add Tuffaha et al., "A Review of Web-Based Tools for Value-of-Information
   Analysis", DOI `10.1007/s40258-021-00662-4`, to support the web-tool field
   account. The review covers SAVI, BCEAweb, RANE, and VICTOR and explains that
   tool choice depends on the decision model and research question.
3. Keep the `value-of-information` comparison. It gives Python readers a useful
   reference point even though its decision problem is narrower.
4. Identify `trd-cea-toolkit` and `vop_poc_nz` as same-author projects when they
   are used as field or impact evidence.
5. Replace mutable repository-root citations for software-use evidence with a
   release, tag, commit, or archived object.

## Prioritised action list

1. **Make the worked example self-contained.** Add the health and cost
   distributions, independence, study target, allocation, outcome variability,
   population, horizon, discounting, study cost, delay, and uptake assumptions.
2. **Define the applied quantities.** Expand QALY, define incremental net
   benefit in one sentence, and explain what "value units" represent.
3. **Bind every software claim to the reviewed release.** Commit and release the
   current EVSI work or remove it from the manuscript; replace all future-tense
   availability language with exact identifiers.
4. **Align the reviewer-facing EVSI documentation with the current callback
   interface.** Provide one complete study-model example that an applied data
   scientist can run.
5. **Correct and strengthen citations.** Fix the Rothery author names, add the
   web-tool review, and use immutable evidence for same-author integrations.
6. **Report results as estimates.** Round simulation results sensibly, include
   their existing bootstrap intervals, identify regression-based EVPPI, and
   distinguish analytical EVSI.
7. **Translate the architecture into research consequences.** Replace
   "provenance", "language-neutral interface", "numerical parity", "backend",
   "likelihood", and "posterior update" where a plain explanation carries the
   same meaning.
8. **Add one bounded cross-domain bridge.** Explain that the same decision
   structure can use uncertain demand, revenue, costs, emissions, or policy
   outcomes, while stating that the demonstrated example is health-economic.
9. **Improve the figure caption and alt text.** Explain the reference lines,
   assumptions, estimation basis, and main result.
10. **Rerun the accessibility review on the official JOSS PDF built from the
    immutable revision.** Check the figure at final size, sentence inventory,
    citations, links, and all reviewer installation paths.

Round 3 should not award accessibility credit for intended releases, planned
documentation, or external-use requests. Credit should be based on the exact
rendered paper and installable revision presented to reviewers.
