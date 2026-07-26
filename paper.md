---
title: "voiage: Value-of-information analysis for decisions about further research"
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
    ror: "0040r6f76"
  - name: "College of Medicine and Public Health, Flinders University"
    index: 2
    ror: "01kpzv902"
  - name: "Centre for Health Policy, The University of Melbourne"
    index: 3
    ror: "01ej9dk98"
date: 26 July 2026
bibliography: paper.bib
repository: https://github.com/edithatogo/voiage
---

# Summary

Decision-makers often need to decide whether existing evidence is enough or a new
study is worth its cost. Value-of-information (VOI) analysis estimates the
benefit of resolving all uncertainty (EVPI), selected uncertainty (EVPPI), or
uncertainty through a proposed study (EVSI). Expected net benefit of sampling
(ENBS) compares a study's expected benefit with its research cost
[@rothery2020voi].

`voiage` calculates these quantities from simulated results or a declared study
model. A versioned analysis record can keep option names, units, uncertain
inputs, and data sources with each result. The same EVPI calculation is
available from Python, R, and Julia. These measures let analysts compare acting
on current evidence with collecting more information, while preserving enough
context for another researcher to interpret the result clearly. In the fixed-seed
health example, EVPI was estimated at 644 value units for each eligible future
person, and perfectly resolving health-gain uncertainty had greater value than
perfectly resolving programme-cost uncertainty.

# Statement of need

A research team may build an uncertainty model in one programming language,
calculate VOI in another, and prepare the report in a third. Moving an analysis
requires more than copying a table of simulated results. Option names, units,
uncertain quantities, population, time horizon, study design, and data sources
affect what an estimate means. If these details are separated from the result,
two identical-looking values may describe different choices, populations, or
periods.

`voiage` validates a versioned description of competing options, uncertain
quantities, units, and input references. Result records link calculations back
to that description; simpler workflows can return a number alone. It is
designed for researchers and analysts using simulation models to decide whether
further evidence is worth collecting in healthcare, public policy, economics,
business, and related fields. Analysts choose the outcomes and value units, so
the structure can describe decisions across these domains. Existing tools
concentrate on methods and reporting within particular software environments;
`voiage` instead provides a shared EVPI calculation and structured reference
for exchanging richer analysis records across languages.

# State of the field

VOI analysis has a long methodological history in decision analysis and health
economics [@claxton1999irrelevance; @ades2004evsi].
The R package `voi` implements several approaches to EVPI, EVPPI, EVSI, and
ENBS [@voi_cran2024]. `BCEA` combines Bayesian cost-effectiveness analysis with
VOI and graphical reporting [@green2022bcea], while `dampack` supports analysis
and visualisation of decision-model outputs, sensitivity analysis, and VOI
[@dampack2024]. These packages provide methods and reporting features absent
from `voiage`'s smaller R and Julia interfaces. The Sheffield Accelerated Value
of Information (SAVI) application provides web-based EVPI and
regression-based EVPPI [@savi2025; @strong2014evppi]. A review of web tools
describes differences in their methods and required inputs
[@tuffaha2021webtools]. Implementation-adjusted methods also estimate how much
value will be realised when new evidence does not fully change practice
[@andronis2016implementation].

Python alternatives address different parts of the problem.
`value-of-information` estimates the value of a noisy signal about one
uncertain option in a simplified binary decision
[@adamczewski2022valueofinformation]. The author's
`trd-cea-toolkit` places VOI within a disease-specific health-economic workflow
[@mordaunt2025trdcea]. `voiage` does not replace these tools or claim broader
method coverage. It instead focuses on exchanging a versioned description of a
decision and calculating EVPI consistently in Python, R, and Julia. Keeping
this shared function in a separate core allows existing specialist packages to
remain independent.

# Software design

One Rust implementation calculates EVPI for Python, R, and Julia, reducing
differences caused by separate language implementations. Analysts still
prepare data, fit models, and plot results in their preferred language. Each
EVPI interface calls Rust directly, so R and Julia users do not need Python.
This choice requires a compiled library for each operating system. Python
currently supports the broadest workflow; R also offers optional Python-backed
EVPPI and EVSI, while Julia supports EVPI only.

Method tests compare specified inputs and outputs with equations implemented
independently, and check repeatability and invalid inputs. Before a method
enters a stable release, it also needs agreement across the Rust implementation
and language interfaces, documentation, release records, and evidence of
platform compatibility. The worked example uses a conjugate normal--normal EVSI
calculation derived from the sampling model in the
[reproduction notes](paper/health-example-methods.md). It assumes equal
allocation, known outcome variability, and a linear relationship between
health gain and net benefit. Ades et al. and Rothery et al. provide the broader
methodological basis for EVSI analysis [@ades2004evsi; @rothery2020voi].

Repository tests compare EVPI with hand-calculated cases in Python and Julia
and with a native Rust-backed case in R. They compare analytical EVSI with a
closed-form reference and reject invalid inputs. Continuous integration tests
Python wheels and native R installation on Linux, macOS, and Windows; Julia is
currently tested on Ubuntu.

# Worked example

The synthetic health example compares a programme with current practice using
10,000 simulated health-gain and programme-cost pairs. Health gain has mean
0.06 quality-adjusted life years (QALYs) and standard deviation 0.03 QALYs;
cost is generated independently with mean 3,000 and standard deviation 650.
Amounts are undated synthetic units with no jurisdictional or payer
perspective. At 50,000 units per QALY, generating expected incremental net
benefit is zero; the fixed sample mean is -15.9 because of simulation error.

The algebraic study illustration estimates the difference in mean QALYs between
two equally allocated groups. Its normal model uses a known standard deviation
of 1.0 QALY only to demonstrate the calculation; it does not specify clinical
follow-up, missing data, or treatment switching and is not a proposed trial.
Total sample sizes range from 50 to 1,200. The study informs health gain, while
programme-cost uncertainty remains in the decision but is not updated.

The programme is preferred in 49.2% of simulations. Resampling gives a 95%
range of 48.2% to 50.2%, which measures simulation error rather than model
uncertainty. Estimated EVPI is 644 units for each eligible future person (624
to 658). Regression-estimated EVPPI is 590 for health gain (569 to 603) and 250
for programme cost (229 to 265). The linear regression is correctly specified
here because net benefit is linear and the inputs are independent; this does
not validate nonlinear or correlated models. Perfectly resolving health-gain
uncertainty has greater value here than perfectly resolving programme-cost
uncertainty. This comparison alone does not identify a preferred study, and
the separate EVPPI estimates should not be added together.

Under the stated assumptions, a 200-participant study has an EVSI of 124 units
per eligible future person. ENBS compares its population value with its cost
[@rothery2020voi]. The calculation assumes 1,300 eligible people annually for
ten years, 3% annual discounting, a fixed study cost of 1.2 million, and 100
units per participant. The discount rate applies to future decision
opportunities, not the generated health and cost outcomes; delay is fixed
independently of sample size. If findings were available immediately and fully
adopted, ENBS is negative at 100 participants and positive at 200. With a
two-year delay and 60% of potential benefit reaching practice, it is negative
at 800 and positive at 1,200. These ranges show where ENBS changes sign; they
do not identify the best sample size. The 60% figure is an assumption, not a
model of how practice changes, and the example does not recommend a real study.

The [sensitivity table](paper/data/synthetic_health_example_sensitivity.csv)
varies outcome variability, population, study cost, delay, and value
realisation. Figure \autoref{fig:health-example} summarises the results. Panel C
shows conditional scenarios, not uncertainty intervals.

![Worked health example. In panel A, the vertical line marks 50,000 value units
per QALY. In panel B, the dashed line marks EVPI. In panel C, markers are
evaluated sizes and connecting lines are visual guides; values above zero
indicate expected population benefit exceeds study cost. EVPI and EVPPI use
10,000 fixed-seed synthetic draws; EVSI is
analytical. All inputs and results are
synthetic.](paper/figures/synthetic_health_example.png){#fig:health-example}

# Research impact statement

A related health-economic project by the same author publishes versioned input
and expected-result formats designed to work with `voiage`
[@vop_poc_nz2026]. This provides a concrete exchange test case, but not evidence
of independent adoption or completed research use. The fixed-seed example,
supporting equations, sensitivity data, and regeneration instructions allow
readers to inspect the calculations. The package has been developed publicly
since July 2025. No completed research-workflow use or engagement by non-authors
has yet been documented.

# AI usage disclosure

Generative artificial intelligence (AI) tools assisted with this work. OpenAI
Codex and Google Jules assisted with code and test generation, refactoring,
documentation, workflows, and manuscript drafting and editing. The current
preparation environment records Codex CLI 0.144.1 and Jules CLI 0.1.42. Codex
used GPT-5-family models; Jules used Google-managed models whose identifiers
were not exposed by the service. Exact model identifiers were not retained for
every historical session. The human author remained the primary decision-maker,
selected the research problem and architecture, thoroughly reviewed, modified,
and validated all retained AI-assisted outputs, and reran the reported tests
and numerical checks. The author accepts responsibility for the software,
manuscript, claims, citations, licensing, and submission. No AI system is an
author.

# Acknowledgements

This work received no external funding. The author declares no competing
interests.

# Software and data availability

The source code and release 1.0.0 are public [@voiage2026]. The fixed-seed
health-example script, `scripts/generate_paper_health_example.py`, and its
machine-readable outputs use synthetic data. The signed v1.0.0 tag is included
in the Software Heritage snapshot
`swh:1:snp:767efde24c97d9f6d730764c1b3bc1a91ba20c32`
[@voiage_software_heritage]. Release 1.0.0 predates the revised EVSI contract
described here. The exact reviewed revision therefore requires a new release
and archive before JOSS submission. Its reproduction manifest records the
inputs, fixed seeds, lockfile digest, output hashes, and verification command.

# References
