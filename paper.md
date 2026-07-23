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

The v1.0.0 release is a hybrid Rust/Python implementation. Rust owns selected
stable aggregation kernels and result contracts. Python provides the broader
public interface, model execution, schema validation, orchestration, reporting,
plotting, and command-line interface. PyO3/Maturin exposes the Python bridge;
the current R and Julia source packages provide direct EVPI bindings through
the shared C ABI and an externally supplied Rust shared library. Wider
cross-language parity is not claimed.

# Statement of need

VOI analysis is central to deciding whether additional research is worthwhile,
but practical implementations are often fragmented across domain-specific
packages, languages, and model formats. This fragmentation makes it difficult
to compare analyses, reproduce results, validate edge cases, and move a model
between a research prototype and an operational decision workflow. Python users
in particular need a maintained, open implementation that supports common VOI
methods while remaining compatible with the broader scientific Python
ecosystem.

`voiage` addresses this need with a common decision-model contract, validated
schemas, deterministic fixtures, explicit diagnostics, reproducible seeds,
machine-readable result envelopes, and a CLI for scripted workflows. The
package is designed for researchers and analysts who need to evaluate the value
of proposed data collection, compare decisions under uncertainty, or integrate
VOI calculations into larger evidence and policy pipelines.

# State of the field

VOI analysis is well established in decision analysis and health economics
[@claxton1999value; @bradley1993expected].
The R ecosystem includes mature packages for particular health-economic
workflows, including Bayesian cost-effectiveness analysis and EVSI methods.
`voiage` complements rather than replaces those packages by providing a
cross-domain decision contract through a hybrid Rust/Python implementation.
Extending one specialist health-economic package would not provide the same
language-neutral contracts or broader problem taxonomy, while independently
reimplementing every method in each language would increase numerical drift.
The project therefore uses shared fixtures and selected Rust kernels, while
retaining specialist packages as preferable where they provide deeper
method-specific validation. Its extension policy separates stable interfaces
from fixture-backed and experimental research surfaces.

# Software design

The stable runtime is organized around validated domain objects, canonical
schemas, selected Rust numerical kernels, diagnostics, provenance, and
adapters. Python still owns broader method paths and user-model orchestration.
The narrow, versioned C ABI has explicit ownership and error transport.
Optional dependencies such as JAX, plotting libraries, and web integrations
are isolated from the base core.

Correctness is supported by language-neutral fixtures, unit and integration
tests, property-based and fuzz tests, metamorphic checks, mutation testing,
coverage gates, Rust diagnostics, ABI checks, benchmark regression budgets, and
clean-install tests. The release process produces platform wheels, a source
archive, checksums, SBOM and provenance evidence, and a signed GitHub release.
Documentation is maintained in Astro/Starlight. The v1.0.0 release is available
from GitHub and PyPI, and the repository records the external conda-forge,
Julia, R, Software Heritage, RRID, and JOSS follow-on gates separately.

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

The author acknowledges the maintainers and developers of NumPy, SciPy, pandas,
xarray, scikit-learn, PyArrow, Polars, Pydantic, Python, Rust, PyO3, Maturin,
R, and Julia. Funding, institutional support, and additional contributors must
be confirmed and added here before submission.

# References

The software citation follows established software-citation guidance
[@smith2016software].
