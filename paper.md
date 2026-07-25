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
  - name: "College of Medicine and Public Health, Flinders University"
    index: 2
  - name: "Centre for Health Policy, The University of Melbourne"
    index: 3
date: 24 July 2026
bibliography: paper.bib
repository: https://github.com/edithatogo/voiage
---

# Summary

Researchers often need to decide whether existing evidence is sufficient or
further research would be worthwhile. Value-of-information (VOI) analysis asks
four related questions: what it would be worth to remove all uncertainty, to
resolve uncertainty about selected inputs, to conduct a proposed study, and,
after study costs, whether that research would provide net value
[@rothery2020voi]. These quantities are expected value of perfect information
(EVPI), expected value of partial perfect information (EVPPI), expected value
of sample information (EVSI), and expected net benefit of sampling (ENBS).

`voiage` calculates EVPI and EVPPI from simulated results, analytical EVSI from
a declared study model, and ENBS from EVSI and research costs. A versioned
specification records option names, units, input identifiers, and data-source
references; typed results link back to it by digest. EVPI uses the same Rust
implementation from Python, R, and Julia; R also offers Python-backed EVPPI and
EVSI, whereas Julia is EVPI-only. In the fixed-seed health example, EVPI was
estimated at 644 value units per person, and uncertainty about health gain had
greater EVPPI than uncertainty about programme cost.

# Statement of need

A research team may build an uncertainty model in one programming language,
calculate VOI in another, and prepare the report in a third. Moving an analysis
requires more than copying a table of simulated results. Option names, units,
uncertain quantities, population, time horizon, study design, and data sources
affect what an estimate means. If these details are separated from the result,
two identical-looking values may describe different choices, populations, or
periods.

`voiage` validates a versioned description of competing options, uncertain
quantities, units, and input references. Typed result records link calculations
back to that description by digest; scalar convenience functions return
numbers without the full context. It is designed for analysts deciding whether
further evidence is worth collecting in healthcare, public policy, economics,
business, and related fields. Analysts choose the outcomes and value units, so
the structure can describe decisions across these domains. `voiage` does not
introduce a new EVPI formula. It provides the same core EVPI calculation across
several languages and a shared, structured reference for exchanging richer
analysis records.

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
method coverage. `voiage` addresses a narrower need: a versioned decision
description and a shared EVPI implementation exposed through Python, R, and
Julia. Adding it to one language-specific package would
make that ecosystem the gateway for the others. A separate core lets specialist
packages remain independent while bindings share a versioned boundary.

# Software design

EVPI is calculated by one Rust implementation, whether the analyst works in
Python, R, or Julia. This reduces the chance that equivalent analyses return
different answers solely because they use different languages. Data
preparation, model fitting, and plotting remain in the analyst's chosen
language. Independent implementations would simplify native packaging but
could drift numerically. Routing every interface through Python would simplify
the bindings but make Python mandatory. Direct native bindings instead preserve
each language's workflow, at the cost of distributing the Rust library
separately. Python supports the broadest workflow; R also provides optional
Python-backed EVPPI and EVSI; and Julia currently supports EVPI.

Scientific checks for a method include specified inputs and outputs,
comparison with equations derived separately from the implementation, and
tests of repeated runs and invalid inputs. Stable promotion additionally
requires cross-language and Rust parity, documentation, release records, and
compatibility evidence. The worked example uses a conjugate normal--normal EVSI
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
10,000 simulated health-gain and programme-cost pairs. Health gain follows a
normal distribution with mean 0.06 quality-adjusted life years (QALYs) and
standard deviation 0.03 QALYs. Cost is generated independently with mean 3,000
and standard deviation 650 value units. At 50,000 value units per QALY, expected
health value equals expected cost, so expected incremental net benefit is zero.
Although mean net benefit is zero by design, information can still improve the
choice.

The stylised study estimates the difference in mean QALYs between two equally
allocated groups. Its normal model uses a known standard deviation of 1.0 QALY
only to demonstrate the calculation; it is not a realistic clinical outcome
model or a proposed trial. Total sample sizes range from 50 to 1,200. The study
informs health gain, while programme-cost uncertainty remains in the decision
but is not updated.

The programme has positive incremental net benefit in 49.2% of simulations.
Its paired 95% percentile-bootstrap interval is 48.2% to 50.2% and describes
Monte Carlo sampling error rather than model uncertainty. Estimated EVPI is
644 value units per person (624 to 658). Regression-estimated EVPPI is 590 for
health gain (569 to 603) and 250 for programme cost (229 to 265). Health-gain
uncertainty therefore has greater potential value to resolve before
considering study feasibility or cost. The EVPPI estimates are separate
conditional analyses, not additive parts of EVPI.

Under the stated assumptions, a 200-participant study has an EVSI of 124 value
units per person. ENBS compares its population value with its cost
[@rothery2020voi]. The calculation assumes 1,300 eligible people annually for
ten years, 3% annual discounting, a fixed study cost of 1.2 million, and 100
value units per participant. With immediate, complete use, ENBS is negative at
100 participants and positive at 200. With a two-year delay and 60% value
realisation, it is negative at 800 and positive at 1,200. These brackets do not
identify an optimum. The value-realisation assumption is not an implementation
model, and the example does not recommend a real study.

The [machine-readable sensitivity table](paper/data/synthetic_health_example_sensitivity.csv)
varies outcome variability, eligible population, fixed study cost, and
scenarios that jointly change delay and value realisation. Figure
\autoref{fig:health-example} summarises these results and scenarios; they do
not assign probabilities to model structures. Panel C shows conditional
scenarios, not uncertainty intervals.

![Worked health example. In panel A, the vertical line marks 50,000 value units
per QALY. In panel B, the dashed line marks EVPI. In panel C, markers are
evaluated sizes and connecting lines are visual guides; values above zero
indicate expected population benefit exceeds study cost. EVPI and EVPPI use
10,000 fixed-seed synthetic draws; EVSI is
analytical. All inputs and results are
synthetic.](paper/figures/synthetic_health_example.png){#fig:health-example}

The [decision-curve table](paper/data/synthetic_health_example_curve.csv),
[summary table](paper/data/synthetic_health_example_summary.csv), and
[study-value table](paper/data/synthetic_health_example_results.csv) provide
the numerical data behind panels A, B, and C, respectively.

# Research impact statement

A related health-economic project by the same author publishes versioned input
and expected-result formats intended for exchange with `voiage`
[@vop_poc_nz2026]. This documents the intended interoperability mechanism, but
not completed use or independent adoption. The fixed-seed example,
implementation-independent equations, sensitivity table, and clean
regeneration allow others to inspect the calculations and outputs. The package
has been developed publicly since July
2025. Attributable non-author engagement or use has not yet been documented.

# AI usage disclosure

Generative artificial intelligence (AI) tools assisted with this work. OpenAI
Codex and Google Jules assisted throughout the software, tests, documentation,
workflows, and manuscript. The current preparation environment records Codex
CLI 0.144.1 and Jules CLI 0.1.42. Codex used GPT-5-family models; Jules used
Google-managed models whose identifiers were not exposed by the service. Exact
model identifiers were not retained for every historical session. The human
author remained the primary decision-maker, selected the research problem and
architecture, reviewed, edited, and validated all retained AI-assisted outputs,
and reran the reported tests and numerical checks. The author accepts
responsibility for the software, manuscript, claims, citations, licensing, and
submission. No AI system is an author.

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
