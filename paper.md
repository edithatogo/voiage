---
title: "voiage: Value of Information Analysis"
tags:
  - value of information
  - decision analysis
  - health economics
  - uncertainty quantification
  - Python
  - Rust
  - reproducible research
authors:
  - given-names: Dylan
    surname: Mordaunt
    affiliation: "1, 2, 3"
    corresponding: true
affiliations:
  - name: "Faculty of Health, Education and Psychology, Victoria University of Wellington"
    index: 1
  - name: "College of Medicine and Public Health, Flinders University"
    index: 2
  - name: "Centre for Health Policy, The University of Melbourne"
    index: 3
date: 22 July 2026
bibliography: paper.bib
repository: https://github.com/edithatogo/voiage
---

# Summary

Value of Information (VOI) analysis estimates how much a decision could improve
if uncertainty were reduced by collecting additional information. It is useful
in health economics, clinical-trial design, public policy, environmental
management, and other settings where decisions must be made before all relevant
quantities are known. `voiage` is an open-source library for computing and
communicating VOI analyses from probabilistic decision models. It provides
implementations of expected value of perfect information (EVPI), expected value
of partial perfect information (EVPPI), expected value of sample information
(EVSI), expected net benefit of sampling (ENBS), cost-effectiveness acceptability
frontiers (CEAF), dominance, and related diagnostics.

Version 1.0.0 uses Rust for selected established calculations. Python supports
the wider range of analyses, including user models, data checks, reporting,
plots, and command-line use. The current R and Julia packages calculate EVPI
directly but do not yet provide the full Python feature set.

# Statement of need

VOI analysis is central to deciding whether additional research is worthwhile,
but practical implementations are often fragmented across domain-specific
packages, languages, and model formats. This fragmentation makes it difficult
to compare analyses, reproduce results, validate edge cases, and move a model
between a research prototype and an operational decision workflow. Python users
in particular need a maintained, open implementation that supports common VOI
methods while remaining compatible with the broader scientific Python
ecosystem.

`voiage` addresses this need with one clear description of the decision,
careful data checks, repeatable examples, visible warnings and assumptions, and
results that can be saved and reviewed. It is designed for researchers and
analysts who need to evaluate the value of proposed data collection, compare
decisions under uncertainty, or include VOI calculations in wider evidence and
policy work.

# State of the field

VOI analysis is well established in decision analysis and health economics
[@claxton1999value; @bradley1993expected].
The R ecosystem includes mature packages for particular health-economic
workflows, including Bayesian cost-effectiveness analysis and EVSI methods.
`voiage` complements rather than replaces those packages by providing a
cross-domain decision framework through a combination of Rust and Python.
Extending one specialist health-economic package would not provide the same
range of decision problems, while rewriting every calculation separately in
each language would make inconsistent results more likely. Specialist packages
remain preferable where they provide deeper support for a particular method.
`voiage` clearly separates established, developing, and experimental features.

# Software design

The software checks that decision data have the expected shape and meaning,
then uses selected Rust calculations or the broader Python methods as
appropriate. Optional features for faster computation, plotting, and web use
are kept separate so that they are not required for a basic analysis.

Correctness is checked with worked examples, unit and integration tests,
generated edge cases, deliberately introduced faults, performance checks, and
fresh installations. The release process also checks package contents,
dependencies, and reproducibility. Version 1.0.0 is available from GitHub and
PyPI; other publication and registry processes are tracked separately.

# Research impact statement

`voiage` is intended to support research in health technology assessment,
clinical-trial design, public policy, environmental decision-making, and
related areas of uncertainty quantification. The repository contains
deterministic analytical and integration fixtures, tutorials, cross-language
contracts, and representative workflows that exercise these use cases. At the
time of preparing this draft, the authors are completing the evidence package
for externally documented research use and adoption; this statement must be
updated with specific public publications, preprints, or research projects
before a JOSS submission if the editorial screening requires them.

# AI usage disclosure

OpenAI Codex and Google Jules assisted with repository analysis, implementation
review, security and release workflow review, documentation drafting, and
preparation of this manuscript. The repository does not retain a complete
retrospective inventory of model versions used across development. Suggestions
were checked against source code, tests, references, and generated artifacts.
The human author remains responsible for the software design, claims,
references, author list, and final text; no AI tool is an author.

# Acknowledgements

The author acknowledges the open-source communities whose statistical,
scientific-computing, and programming tools support this work. Funding,
institutional support, and additional contributors must be confirmed before
submission.

# References

The software citation follows established software-citation guidance
[@smith2016software].
