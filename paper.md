---
title: "voiage: Value of Information Analysis with a Rust Core and Polyglot Bindings"
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
    affiliation: 1
    corresponding: true
affiliations:
  - name: "Independent Researcher, Australia"
    index: 1
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

The v1.0.0 release combines a Python user interface with an authoritative Rust
execution core. Python provides the public façade, schema validation,
orchestration, reporting, plotting, and command-line interface. Rust provides
the stable numerical kernels and shared result contracts; PyO3/Maturin exposes
the Python bridge, while the R and Julia bindings use the shared C ABI. This
architecture makes numerical behavior explicit and testable across languages
without requiring users to install a large systems stack.

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
Python-first, cross-domain interface and by making the execution contract
portable to Rust, R, and Julia. The project emphasizes shared fixtures and
cross-language conformance rather than a separate numerical implementation in
each binding. Its extension policy also separates stable methods from optional
and experimental research surfaces.

# Software design

The stable runtime is organized around validated domain objects, canonical
schemas, numerical kernels, diagnostics, provenance, and adapters. The Rust
workspace contains the authoritative numerical implementation and a narrow,
versioned C ABI with explicit ownership and error transport. The Python package
uses PyO3 and Maturin for its native extension and retains only the façade and
workflow responsibilities needed by users. R and Julia consume the same ABI;
their packages do not reimplement VOI policy. Optional dependencies such as
JAX, plotting libraries, and web integrations are isolated from the base core.

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

Generative AI tools assisted with repository analysis, security and release
workflow review, documentation drafting, and preparation of this manuscript.
The human author remains responsible for the software design, claims,
references, author list, and final text. All AI-assisted changes are subject to
human review and repository tests; no AI tool is an author or a substitute for
the human author’s scientific and editorial decisions.

# Acknowledgements

The author acknowledges the maintainers and developers of NumPy, SciPy, pandas,
xarray, scikit-learn, PyArrow, Polars, Pydantic, Python, Rust, PyO3, Maturin,
R, and Julia. Funding, institutional support, and additional contributors must
be confirmed and added here before submission.

# References

The software citation follows established software-citation guidance
[@smith2016software].
