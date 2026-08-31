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
date: 30 August 2026
bibliography: paper.bib
repository: https://github.com/edithatogo/voiage
---

# Summary

Decision-makers weigh acting on current evidence against commissioning a study.
Value-of-information (VOI) analysis
estimates the expected improvement in a decision from better information.
Expected value of perfect information (EVPI) measures the value of resolving
all modelled uncertainty; expected value of partial perfect information (EVPPI)
measures the value of resolving selected inputs; and expected value of sample
information (EVSI) measures the value of a proposed study. Expected net benefit
of sampling (ENBS) compares population EVSI with research cost
[@rothery2020voi].

`voiage` calculates EVPI and EVPPI from simulated net-benefit samples, estimates
EVSI from a declared study model, and combines study value and cost for ENBS. A
versioned record preserves option names, units, uncertain inputs, and data
sources with each result. Python, R, and Julia provide native EVPI and ENBS. In
the fixed-seed health example, EVPI was estimated at 644 value units—a generic
scale—for each person affected by later decisions during the horizon. It supports
comparisons of
further-research choices.

# Statement of need

A research team may build an uncertainty model in one programming language,
calculate VOI in another, and prepare the report in a third. Moving an analysis
requires more than copying a table of simulated results. Option names, units,
uncertain quantities, population, time horizon, study design, and data sources
affect what an estimate means. If these details are separated from the result,
two identical-looking values may describe different choices, populations, or
periods.

`voiage` validates a versioned description of competing options, uncertain
quantities, units, and input references, then links each calculation to that
description. It is designed for researchers and analysts using simulation
models to decide whether further evidence is worth collecting. The same
structure could describe a healthcare programme, policy intervention, business
investment, or marketing strategy when the analyst defines the alternatives,
uncertain outcomes, affected population, and value scale. The worked example
and current methodological validation are health-economic. `voiage` therefore
contributes a shared EVPI calculation and a portable record of decision
context, rather than the broader method and reporting coverage of established
specialist tools.

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

Python alternatives differ.
`value-of-information` estimates the value of a noisy signal about one
uncertain option in a simplified binary decision
[@adamczewski2022valueofinformation]. The author's
`trd-cea-toolkit` places VOI within a disease-specific health-economic workflow
[@mordaunt2025trdcea]. `voiage` does not replace these tools or claim broader
method coverage. Its contribution combines matching EVPI and ENBS calculations
across Python, R, and Julia with Python decision records. Standalone native
interfaces avoid making Python mandatory for R and Julia. Python retains the decision
record; R and Julia expose scalar calculations checked against shared fixtures.
Specialist tools retain their broader methods and reporting.

# Software design

Python and Julia use the main Rust core; R bundles a separate, dependency-free
Rust kernel for EVPI and ENBS. Shared numerical fixtures check agreement across
these implementations. R and Julia users do not need Python for these methods.
The trade-off is installation: each operating system needs compatible compiled
code. R builds its kernel offline from the source archive; Julia requires a
separately supplied native library. Python supports the broadest workflow; R
also offers optional Python-backed EVPPI and EVSI.

Method tests compare specified inputs and outputs with separately implemented
reference equations, and check repeatability and invalid inputs. A method is
labelled stable only after its implementation, exposed language interfaces,
documentation, release record, and supported platforms agree. The worked
example uses a normal model in which an equally allocated two-group study
updates uncertainty about average health gain. It assumes known outcome
variability and a linear relationship between health gain and net benefit; the
[reproduction notes](paper/health-example-methods.md) give the equations. Ades
et al. and Rothery et al. describe the broader EVSI methodology
[@ades2004evsi; @rothery2020voi].

Repository tests compare EVPI with hand-calculated cases in Python and Julia
and with a native Rust-backed case in R. They compare analytical EVSI with a
closed-form reference and reject invalid inputs. Continuous-integration
workflows are configured to test Python wheels and the native R and Julia
interfaces on Linux, macOS, and Windows.

# Worked example

The synthetic health example compares a programme with current practice using
10,000 simulated health-gain and programme-cost pairs. Health gain has mean
0.06 quality-adjusted life years (QALYs) and standard deviation 0.03 QALYs;
cost is generated independently with mean 3,000 and standard deviation 650.
Amounts are undated synthetic units with no jurisdictional or payer
perspective. At 50,000 units per QALY, the generating distribution has an
expected incremental net benefit of zero. The fixed sample mean is -15.9 units
because a finite random sample does not reproduce that expectation exactly.

The algebraic illustration estimates the difference in mean QALYs between two
equally allocated groups. Its normal model uses a known standard deviation of
1.0 QALY only to demonstrate the calculation; it is not a proposed trial.
Total sample sizes range from 50 to 1,200. The study updates health gain, not
programme-cost uncertainty.

The programme is preferred in 49.2% of simulations. Repeatedly resampling the
10,000 paired simulations gives 95% Monte Carlo intervals of 48.2% to 50.2%
for programme preference, 624 to 658 for EVPI, 569 to 603 for health-gain
EVPPI, and 229 to 265 for programme-cost EVPPI. These intervals measure
numerical variation caused by a finite simulation sample; they do not represent
uncertainty about whether the model is correct. Estimated EVPI is 644 units;
EVPPI is 590 for health gain and 250 for programme cost. The regression is
appropriate because net benefit is linear and the inputs were generated
independently; it does not validate nonlinear or correlated applications. This
comparison does not identify a preferred study, and EVPPI estimates should not
be added together.

Under the stated assumptions, a 200-participant study has an EVSI of 124 units
per eligible future person. ENBS multiplies that value across expected
beneficiaries and subtracts study cost [@rothery2020voi]. The calculation
assumes 1,300 eligible people annually for ten years, 3% annual discounting, a
fixed study cost of 1.2 million, and 100 units per participant. Immediate,
complete use changes ENBS from negative at 100 participants to positive at 200.
With a two-year delay and 60% realisation, it changes from negative at 800 to
positive at 1,200. These brackets do not identify the best sample size; delay
and realisation are assumptions, not predictions of study conduct or practice
change.

The [sensitivity table](paper/data/synthetic_health_example_sensitivity.csv)
varies assumed individual study-outcome standard deviation, population, study
cost, delay, and value realisation. \autoref{fig:health-example} summarises the
results. Panel C
shows conditional scenarios, not uncertainty intervals.

![Three-panel synthetic health example. Panel A shows how the probability of
preferring the programme changes with the value assigned to one QALY; the
reference value is 50,000 units per QALY. Panel B shows that resolving
uncertainty about health gain has greater estimated value than resolving
uncertainty about programme cost, while both remain below EVPI. Panel C shows
ENBS at six evaluated sample sizes under immediate full use and delayed partial
use of study findings. Values above zero mean that expected population benefit
exceeds study cost. Connecting lines are visual guides, not estimates between
evaluated sizes. EVPI and EVPPI use 10,000 fixed-seed synthetic draws; EVSI is
analytical. All inputs and results are synthetic.](paper/figures/synthetic_health_example.png){#fig:health-example}

# Research impact statement

Historical developer research use calculated EVPI for the same author's
HPV-vaccination model [@vop_poc_nz2026]. An automated v2.2.0 replay reproduced
its 500 draws and EVPI through a hash-checked CSV handoff between separate
supported environments. This is not a combined installation, additional human
use, independent adoption, or a policy estimate. The supplied equations,
sensitivity data, and regeneration commands allow others to check these
bounded calculations. Development has been public since July 2025; engagement
by non-authors has not yet been documented.

# AI usage disclosure

OpenAI Codex and Google Jules assisted with code and test generation,
refactoring, documentation, workflows, and manuscript drafting and editing.
Historical preparation records list Codex CLI 0.144.1 and Jules CLI 0.1.42.
Codex used GPT-5-family models; Jules used Google-managed models. Exact model
identifiers were not retained for every session. Codex assisted with the
August release-evidence and manuscript repairs. The human author was the
primary decision-maker and confirmed on 27 July 2026 that all retained
AI-assisted outputs had been reviewed and validated. That historical
attestation does not cover these
later revisions; final human review remains pending. The author retains responsibility
for accuracy, claims, citations, licensing, and submission. No AI system is an
author.

# Acknowledgements

This work received no external funding. The author declares no competing
interests.

# Software and data availability

The source code, signed release 2.2.0, and synthetic worked-example materials
are public [@voiage2026]. The cited Software Heritage snapshot preserves
v2.0.0, not v2.2.0 [@voiage_software_heritage]. Repository manifests bind the
current release's checksums, provenance and software bill of materials.
Historical worked-example manifests record fixed seeds, input and output
hashes, execution environments, and the original regeneration command. Current
CI rechecks the portable outputs against those retained files. All
worked-example data are synthetic; the repository supplies generation scripts
and equations. A v2.2.0 Software Heritage snapshot and DOI-bearing archive
remain unconfirmed.

# References
