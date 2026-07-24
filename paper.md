---
title: "voiage: Value of Information for Research, Implementation, and Decision Making"
tags:
  - value of information
  - decision analysis
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
date: 22 July 2026
bibliography: paper.bib
repository: https://github.com/edithatogo/voiage
---

# Summary

Value of Information (VOI) analysis estimates how much a decision could improve
if uncertainty were reduced by collecting additional information. It is useful
in health economics, clinical-trial design, public policy, environmental
management, and other settings where decisions must be made before all relevant
quantities are known. `voiage` computes and communicates VOI analyses from
probabilistic decision models. It provides expected value of perfect
information (EVPI), expected value of partial perfect information (EVPPI),
expected value of sample information (EVSI), expected net benefit of sampling
(ENBS), cost-effectiveness acceptability frontiers, dominance checks, and
diagnostics.

Version 1.0.0 uses binding-independent Rust components for shared domain
contracts, diagnostics, serialization, and selected established numerical
kernels. Python provides the wider modelling, validation, reporting, plotting,
and command-line surface. The current R and Julia source packages call the same
versioned Rust interface for EVPI but do not yet provide the full Python feature
set or standalone language-registry installation. This boundary is documented
so that users can distinguish a shared calculation from a language-specific
convenience function.

# Statement of need

VOI analysis supports decisions about whether additional research is worthwhile,
but practical implementations are often fragmented across domain-specific
packages, languages, and model formats. This fragmentation makes it difficult
to compare analyses, reproduce results, validate edge cases, and move a model
between a research prototype and an operational decision workflow. It also
creates a risk that the same mathematical quantity is interpreted differently
after data have been reshaped, relabelled, or transferred between tools.

`voiage` addresses this need with one clear description of the decision,
careful data checks, repeatable examples, visible warnings and assumptions, and
results that can be saved and reviewed. It is designed for researchers and
analysts who need to evaluate the value of proposed data collection, compare
decisions under uncertainty, or include VOI calculations in wider evidence and
policy work. Python users receive an installable scientific-computing interface,
while the shared Rust contracts provide a route for R and Julia analyses to use
the same validated inputs and calculation rules rather than independent
reimplementations.

# State of the field

VOI analysis is well established in decision analysis and health economics
[@claxton1999irrelevance; @ades2004evsi].
The R package `voi` provides EVPI, EVPPI, EVSI, and ENBS through several
computational methods [@voi_cran2024], while `BCEA` provides a mature Bayesian
cost-effectiveness workflow and graphical analysis [@green2022bcea]. SAVI
offers a focused web workflow for rapid analysis of probabilistic sensitivity
analysis output and regression-based EVPPI [@strong2014evppi]. These tools are
appropriate choices when their specialised health-economic workflows match the
research question.

`voiage` does not seek to replace them. Its separate package is justified by a
different scope: a language-neutral decision/result contract, explicit
provenance and maturity labels, cross-domain decision problems, and one Rust
calculation boundary that can be exercised by multiple language bindings.
Adding those contracts and non-health domains to a specialist R package would
change that package's purpose; separately reimplementing each calculation in
Python, R, and Julia would weaken parity. The trade-off is that `voiage` has less
depth than established specialist tools for some methods, and its non-Python
bindings currently expose a narrower stable surface.

# Software design

The design separates mathematical kernels from language-facing orchestration.
Rust owns binding-independent types and selected stable reductions; thin PyO3
and C interfaces expose those operations without placing plotting, data-frame,
or language-runtime dependencies in the numerical crates. Python retains
labelled data, user-model orchestration, and methods that have not yet met the
cross-language promotion criteria. R and Julia use the versioned C boundary for
their released EVPI path.

This architecture trades a larger repository and explicit interface contracts
for consistent behaviour across languages. It also permits reviewers to test
the kernels without a Python runtime and to test the Python wheel as a normal
user would. Optional plotting, accelerator, and web dependencies are separated
from the basic installation. Stable, developing, and experimental methods are
labelled independently because implementation alone does not establish that a
method is suitable for every decision problem.

Correctness evidence includes analytical fixtures, unit and integration tests,
property-based and differential tests, mutation testing, Rust fuzzing and Miri,
cross-language conformance fixtures, clean installations, and cross-platform
continuous integration. The v1.0.0 release includes source and platform wheels,
checksums, a software bill of materials, and provenance attestations.

# Research impact statement

The repository records two concrete developer-led research uses. First, the
synthetic health study accompanying the preprint uses `voiage` to compare
parameter uncertainty and alternative study sizes under stated uptake, delay,
population, and cost assumptions. The fixed-seed script, plotted results, and
machine-readable outputs allow the calculation to be rerun and inspected.
Second, a versioned integration contract with the `vop_poc_nz` research
repository carries decision-model records, provenance, schema fingerprints, and
expected numerical results between the projects. These are public,
reproducible materials rather than evidence of independent adoption.

The package has been developed publicly since July 2025. Its release and
documentation support further use in health technology assessment, trial
design, public policy, and other probabilistic decision models, but claims of
external uptake will be added only when attributable evidence is available.

# AI usage disclosure

OpenAI Codex using GPT-5-family models and Google Jules using
service-managed models assisted with repository analysis, code and test
drafting, refactoring, documentation, security and release-workflow review, and
manuscript drafting and copy-editing. Exact model identifiers were not retained
for every historical session, and Google Jules did not expose a stable model
version in the repository record. The human author selected the research
problem and architecture, reviewed and edited merged changes, and validated
AI-assisted outputs against source code, tests, references, numerical fixtures,
and generated artefacts. The human author remains responsible for the
software, claims, citations, authorship, and final submission; no AI system is
an author. AI tools will not be used to compose substantive exchanges with JOSS
editors or reviewers.

# Acknowledgements

The author acknowledges the maintainers of the statistical,
scientific-computing, packaging, and research-software infrastructure on which
this work depends. No external funding is declared for this submission.

# References

Version 1.0.0 is available from the public release [@voiage2026] and the
repository snapshot is preserved by Software Heritage
[@voiage_software_heritage]. The citation metadata follows established
software-citation guidance [@smith2016software].
