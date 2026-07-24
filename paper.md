---
title: "voiage: Value of Information for Research, Implementation, and Decision Making"
tags:
  - value of information
  - decision analysis
  - research prioritisation
  - decision support
  - health economics
  - uncertainty quantification
  - Python
  - Rust
  - R
  - Julia
authors:
  - given-names: Dylan
    surname: Mordaunt
    affiliation: "1, 2, 3"
    corresponding: true
    orcid: 0000-0002-9775-0603
affiliations:
  - name: "Faculty of Health, Education and Psychology, Victoria University of Wellington"
    index: 1
  - name: "College of Medicine and Public Health, Flinders University"
    index: 2
  - name: "Centre for Health Policy, The University of Melbourne"
    index: 3
date: 24 July 2026
bibliography: paper.bib
repository: https://github.com/edithatogo/voiage
---

# Summary

Researchers often choose between options before all uncertainty can be
resolved. Value of information (VOI) analysis asks whether better information
could improve that choice enough to justify its cost [@rothery2020voi].
Expected value of perfect information (EVPI) measures the expected cost of
deciding with current uncertainty. Expected value of partial perfect
information (EVPPI) measures the value of resolving selected inputs. Expected
value of sample information (EVSI) measures a proposed study's expected
benefit, while expected net benefit of sampling (ENBS) subtracts its cost.

`voiage` calculates these measures from simulations of uncertain decisions
while retaining the alternatives, units, assumptions, warnings, and input
provenance needed to interpret the results. Python provides the broadest
workflow. A Rust core shares selected calculations with narrower R and Julia
interfaces. In the synthetic health example, estimated EVPPI was larger
for health gain than for programme cost, and a scenario combining delay with
incomplete realisation of information value required a larger study to produce
positive ENBS.

# Statement of need

Researchers use VOI to assess whether collecting more evidence is likely to
improve a decision enough to justify its cost. An analysis may use specialist
packages and web tools written in different programming languages, with model
outputs supplied in several formats. Moving an analysis between them requires
more than transferring numbers. Strategy names, units, groups of uncertain
inputs, population, time horizon, implementation, study design, and data
sources all affect interpretation. Losing this information can make nominally
identical results describe different decisions.

`voiage` stores the decision description with its calculated results. It checks
that required fields and data dimensions are consistent. The package is
intended for researchers and analysts comparing choices under uncertainty,
prioritising research, or incorporating VOI into a wider evidence assessment.
Decision descriptions accept analyst-supplied outcome and value units, so
applications are not limited to health outcomes. For example, the same decision
structure could represent uncertain demand, revenue, emissions, or policy
outcomes, although the demonstrated example below is in health economics.
Existing VOI tools provide deeper specialist methods. `voiage` focuses on
preserving the meaning of decisions and selected results when a research
workflow uses more than one language.

# State of the field

VOI analysis is well established in decision analysis and health economics
[@claxton1999irrelevance; @ades2004evsi].
The R package `voi` provides several methods for EVPI, EVPPI, EVSI, and ENBS
[@voi_cran2024]. `BCEA` combines cost-effectiveness analysis with graphical
reporting [@green2022bcea], and `dampack` supports decision-model analysis,
sensitivity analysis, visualisation, and VOI [@dampack2024]. The Sheffield
Accelerated Value of Information (SAVI) application provides a web interface
for EVPI and regression-based EVPPI [@savi2025; @strong2014evppi]. A review of
web-based VOI tools discusses SAVI and related applications, including their
input and method differences [@tuffaha2021webtools]. These tools remain
appropriate when their methods, language, and reporting conventions fit the
research question.
Implementation-adjusted EVSI methods can model how new evidence changes
adoption rather than applying a constant multiplier to population value
[@andronis2016implementation].

Python also has specialised alternatives: `value-of-information` analyses a
single uncertain option observed through a noisy signal
[@adamczewski2025valueofinformation], while the author's `trd-cea-toolkit`
embeds VOI in a disease-specific health-economic workflow
[@mordaunt2025trdcea].

The tools reviewed here provide deeper method-specific analyses within R,
Python, or web workflows. Their documentation does not describe a shared
contract carrying strategy names, units, assumptions, provenance, and selected
calculations across Python, R, and Julia. Adding that contract to one package
would leave the cross-language problem unresolved. `voiage` makes the contract
its main design boundary. Its shared EVPI calculation lets the three interfaces
interpret the same decision and return the same result. The trade-off is less
method-specific depth and narrower R and Julia interfaces than Python provides.

# Software design

The design separates calculations that should give the same answer in every
language from tasks that differ between languages, such as loading data,
fitting models, and making plots. Rust holds the shared data rules and selected
calculations. Python provides the most complete workflow, while R and Julia
currently expose the shared EVPI calculation. This arrangement reduces the
risk that the same calculation will be interpreted differently when
researchers move between languages. R and Julia currently require users to
provide a platform-specific build of the native library.

A method is labelled stable only when its inputs and outputs are specified, its
results agree with a calculation made independently of the implementation, and
tests cover repeatability and invalid inputs. The documentation also says when
a result is approximate or requires optional software. A working
implementation therefore does not imply that an estimator is suitable for
every research question. The analytical EVSI calculation uses a normal prior
for incremental health gain and a two-arm normal sampling model with equal
allocation, known common outcome variability, and a linear relationship
between health gain and incremental net benefit
[@ades2004evsi; @rothery2020voi]. For total sample size \(n\), the difference
between the two sample means has variance \(4\sigma^2/n\).

The generic two-loop estimator fits one multivariate normal distribution to
correlated quantities, then uses it to simulate study data and update the
decision. It retains negative Monte Carlo estimates as diagnostics rather than
replacing them with zero. Analysts can repeat runs, increase simulation size,
assess convergence, and check a custom study model. The analytical route is
stable; generic and compatibility estimators require method-specific
validation.
Implementation-adjusted methods are also developing and are not used in the
worked example.

The current repository tests calculations against known results, rejects
invalid data, checks repeatability across implementations, and exercises clean
installations on supported operating systems. Release 1.0.0 contains a source
distribution, three platform wheels, and checksums. The release record does not
contain the analytical EVSI implementation or the revised generic EVSI
contract described here.

# Worked example

The synthetic health example compares a programme with current practice when
health gain and programme cost are uncertain. Across 10,000 generated draws,
incremental health gain was generated from a normal
distribution with a mean of 0.06 quality-adjusted life years (QALYs) and a
standard deviation of 0.03 QALYs. Incremental programme cost was generated
independently from a normal distribution with a mean of 3,000 and a standard
deviation of 650 value units. Here, value units denote the common
monetary-equivalent scale specified for the decision. At 50,000 value units per
QALY, incremental net benefit is the value of the health gain minus programme
cost.

The stylised study estimates the difference in mean QALYs between two equally
allocated arms. Individual outcomes are assumed normal with a known common
standard deviation of 1.0 QALY, and candidate total sample sizes range from 50
to 1,200. The study informs health gain; programme-cost uncertainty remains in
the decision model but is not updated.

Across the 10,000 draws, the programme has positive incremental net benefit in
49.2% of simulations (48.2% to 50.2%).
Paired 95% percentile-bootstrap intervals, based on 1,000 resamples with a
fixed seed, quantify Monte Carlo uncertainty in the estimates rather than
uncertainty about the specified synthetic model. The simulation and bootstrap
seeds are 20260723 and 20260724. Estimated EVPI is 644 value
units per person (624 to 658); regression-based EVPPI is 590 value units per
person for health gain (569 to 603) and 250 value units per person for
programme cost (229 to 265).

For a total sample of 200, the analytical study model gives an EVSI of 124 per
person. ENBS multiplies per-person EVSI by the discounted number of eligible
decision opportunities and subtracts study cost [@rothery2020voi]. Benefits
are assumed to accrue at the end of each year, while study costs occur at time
zero. ENBS assumes 1,300 eligible people per year for ten years, 3% annual
discounting, a fixed study cost of 1.2 million, and a cost of 100 per
participant. It changes sign between 100 and 200 participants when information
is available immediately and all eligible decisions realise its value. In the
delayed scenario, benefits begin in year three and 60% of eligible decisions
are assumed to realise the value of information; ENBS changes sign between 800
and 1,200 participants. This is a reduced-form realisation assumption rather
than an explicit model of intervention uptake or implementation-adjusted EVSI.
The example shows how study size, the timing of evidence, and the assumed
proportion of information value realised affect ENBS; it is not an estimate
for a real trial.

The machine-readable sensitivity table,
`paper/data/synthetic_health_example_sensitivity.csv`, varies outcome
variability, eligible population, fixed study cost, and paired
delay/value-realisation scenarios. These scenarios illustrate dependence on
assumptions; they do not form a probabilistic analysis of structural
uncertainty.

![Worked health example. The programme has positive incremental net benefit in
about half of simulations at 50,000 value units per QALY. Estimated EVPPI is
larger for health gain than for programme cost. Delaying the availability of
evidence and reducing its realised value shifts positive ENBS to larger sample
sizes. In panel A, the vertical line marks 50,000 value units per QALY. In
panel B, the dashed line marks EVPI. In panel C, values above zero indicate
that expected population benefit exceeds study cost. EVPI and EVPPI use 10,000
fixed-seed synthetic draws; EVSI is analytical. All inputs are
synthetic.](paper/figures/synthetic_health_example.png)

# Research impact statement

The same-author Value of Perspective Proof of Concept Aotearoa New Zealand
(`vop_poc_nz`) health-economic workflow publishes a versioned bundle of
schemas, fixtures, source records, and expected results for exchange with
`voiage` [@vop_poc_nz2026]. This demonstrates an interoperability contract, not
independent adoption. The package has been developed publicly since July 2025;
attributable non-author use has not yet been documented.

# AI usage disclosure

Generative artificial intelligence (AI) tools assisted with this work. OpenAI
Codex, using GPT-5-family models, and Google Jules, using service-managed
models, assisted with repository analysis, code and test drafting, refactoring,
documentation, workflow review, and manuscript editing. Exact model identifiers
were not retained for every historical session. The human author selected the
research problem and architecture, reviewed this manuscript, and validated the
reported code, references, and numerical results against repository tests and
generated evidence. The author accepts responsibility for this manuscript, the
software, its claims and citations, and the submission; no AI system is an
author.

# Acknowledgements

This work received no external funding. The author declares no competing
interests.

# Software and data availability

The Python package and release 1.0.0 are public [@voiage2026]. The fixed-seed
health-example script, `scripts/generate_paper_health_example.py`, and its
machine-readable outputs use synthetic data. The repository is preserved by
Software Heritage as
`swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`
[@voiage_software_heritage]. Release 1.0.0 predates the revised EVSI contract
described here. The submitted paper will cite a release made from the exact
reviewed revision, together with its release-evidence manifest.

# References
