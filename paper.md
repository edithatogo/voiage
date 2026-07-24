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

Value of Information (VOI) analysis estimates the expected improvement in a
decision outcome that could follow from resolving uncertainty before a choice
is made [@rothery2020voi]. It asks four
practical questions: could uncertainty change the preferred choice; which
uncertain quantities matter most; how much would a proposed study improve the
decision; and would that improvement be worth the study's cost? These
questions correspond to the expected value of perfect information (EVPI),
expected value of partial perfect information (EVPPI), expected value of sample
information (EVSI), and expected net benefit of sampling (ENBS).

`voiage` calculates these measures from simulated decision-model results. Its
decision and result records can retain the alternatives, units, assumptions,
warnings, and a record of where the inputs came from. Python is the broadest
interface. Rust provides the shared input rules and selected
calculations, including EVPI and EVSI for a two-arm study with a normal prior
and normal likelihood. The R and Julia source packages expose a narrower EVPI
interface. Readers can therefore distinguish calculations shared through Rust
from those available only through a particular language interface.
In the synthetic health example, uncertainty about health gain accounted for
more decision value than uncertainty about programme cost, and a scenario
combining delayed and partial uptake required a larger study to produce
positive net benefit.

# Statement of need

Analysts use VOI to decide whether proposed data collection is worthwhile, but
the calculations may span specialist packages, programming languages, web
tools, and model-output formats. Moving an analysis between them requires more
than transferring a numerical array. A result depends on strategy names,
units, and parameter groups as well as its numerical values. Population, time
horizon, implementation, study design, and provenance determine how that result
should be interpreted. Losing this information can make two nominally identical
results describe different decisions.

`voiage` provides inspectable decision and result objects. Its structured
calculation interfaces validate the fields and array dimensions they use. The
package is intended for researchers and analysts comparing choices under
uncertainty, prioritising research, or incorporating VOI into a wider evidence
assessment.
Decision descriptions accept analyst-supplied outcome and value units, so
applications are not limited to health outcomes. For example, the same decision
structure could represent uncertain demand, revenue, emissions, or policy
outcomes, although the demonstrated example below is in health economics.
Existing VOI tools provide deeper method-specific workflows; `voiage` instead
focuses on keeping the decision description and selected results consistent
when a research workflow uses more than one language.

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

Python also has specialised alternatives: `value-of-information` analyses a
single uncertain option observed through a noisy signal
[@adamczewski2025valueofinformation], while the author's `trd-cea-toolkit`
embeds VOI in a disease-specific health-economic workflow
[@mordaunt2025trdcea].

`voiage` instead keeps the decision description, assumptions, source record,
and selected results together. Its shared EVPI contract allows the Python, R,
and Julia interfaces to interpret the same decision and return the same
calculation. The project was developed separately because its research
requirement is a language-neutral decision-and-result contract, whereas the
compared tools provide method-specific R or web workflows. This contract
supports workflows that combine language interfaces. The trade-off is less
method-specific depth than established R packages and substantially narrower R
and Julia interfaces than the Python interface.

# Software design

The design separates calculations intended to be shared from data handling,
modelling, plotting, and reporting that remain language-specific. Rust
implements common types, validation rules, and selected calculations. Python
provides the broader workflow, including labelled data and user-defined models.
The R and Julia packages call the shared EVPI calculation but do not reproduce
the full Python interface. Sharing the calculation avoids duplicating it, but
each language package still needs installation tests and checks that it returns
the same result.

A method is labelled stable only when its inputs and outputs are specified, its
results agree with a calculation made independently of the implementation, and
tests cover repeatability and invalid inputs. The documentation also says when
a result is approximate or requires optional software. A working
implementation therefore does not imply that an estimator is suitable for
every research question. The analytical EVSI calculation applies to an equally
allocated two-arm study, normally distributed current uncertainty and study
outcomes, known outcome variability, and a linear relationship between the
outcome and net benefit. These assumptions determine how the study changes
uncertainty. The built-in two-loop model updates correlated uncertain
quantities together under one fitted multivariate-normal prior: current value,
possible study results, and posterior value are evaluated under that same
prior. This developing estimator uses genuine Gaussian Monte Carlo draws and
returns the untruncated estimate. Because EVSI is nonnegative, a negative
estimate is retained as a diagnostic rather than silently replaced with zero.
Analysts can repeat the calculation with different seeds, increase the
simulation size, assess convergence, and check the coherence of any custom
study model.
Analysts can supply the study simulation and corresponding joint update for
other study models. The analytical model is the stable study-specific route;
the generic two-loop and compatibility estimators are not labelled stable
without method-specific validation.

The current repository tests calculations against known results, rejects
invalid data, checks repeatability across implementations, and exercises clean
installations on supported operating systems. Release 1.0.0 contains a source
distribution, three platform wheels, and checksums. The release record does not
contain the analytical EVSI implementation or the revised generic EVSI
contract described here.

# Research impact statement

The repository contains one synthetic worked example and one same-author
interoperability bundle for another research workflow. The health example
compares a programme with current practice when health effect and programme
cost are uncertain. Across 10,000 independently generated draws, incremental
health gain has a Normal distribution with mean 0.06 quality-adjusted life
years (QALYs) and standard deviation 0.03 QALYs, while incremental programme
cost has an independent Normal distribution with mean 3,000 and standard
deviation 650 value units. In this synthetic example, value units represent a
common currency-equivalent scale supplied by the decision maker. At 50,000
value units per QALY, incremental net benefit is the value of the health gain
minus programme cost. The proposed study informs health gain, not programme
cost. It has equal allocation, individual outcome standard deviation 1.0 QALY,
and total sample sizes from 50 to 1,200.

The programme is preferred in 49.2% of the fixed-seed simulations (bootstrap
95% interval 48.2% to 50.2%). Estimated EVPI is 644 value units per person
(bootstrap 95% interval 624 to 658); regression-based EVPPI is 590 for health
gain (569 to 603) and 250 for programme cost (229 to 265), with 95% bootstrap
intervals in parentheses. For a total sample of 200, the analytical study model
gives an EVSI of 124 per person. ENBS assumes 1,300 eligible people per year for
ten years, 3% annual discounting, a fixed study cost of 1.2 million, and a cost
of 100 per participant. With immediate full uptake, ENBS is negative at 100
participants and positive at 200. With a two-year delay and 60% uptake, it is
negative at 800 participants and positive at 1,200. The synthetic example
illustrates how study size and implementation assumptions affect estimated
research value; it is not an estimate for a real trial.

![Worked health example. The programme is preferred in about half of
simulations at 50,000 value units per QALY; uncertainty about health gain has
greater value than uncertainty about programme cost; and delayed, partial
uptake requires a larger study before expected population benefit exceeds
study cost. In panel A, the vertical line marks 50,000 value units per QALY. In
panel B, the dashed line marks the value of resolving all uncertainty. In panel
C, values above zero indicate that expected population benefit exceeds study
cost. EVPI and EVPPI use 10,000 fixed-seed synthetic draws; EVSI is analytical.
All inputs are synthetic.](paper/figures/synthetic_health_example.png)

The developer-led Value of Perspective Proof of Concept Aotearoa New Zealand
(`vop_poc_nz`) health-economic research workflow contains a versioned
interoperability bundle of schemas, fixtures, source records, and expected
results for exchange with `voiage` [@vop_poc_nz2026]. The bundle was created
by the same author and is not evidence of independent adoption. The package has
been developed publicly since July 2025. Attributable non-author use has not
yet been documented.

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
