# Track Specification: External VOI Library Feature Parity

## Overview

Create a reproducible landscape of VOI software and implement justified
open-source features in VOIAGE without a runtime dependency on competitors.

## Requirements

1. Search CRAN, R-universe, PyPI, crates.io, Julia General, Mojo channels,
   GitHub/GitLab, literature supplements, public web tools, and commercial
   documentation at least quarterly and before minor releases.
2. Seed and verify R `voi`, BCEA, SAVI, Analytica VOI behavior, Pyro OED, and
   BoTorch acquisition functions.
3. Record versions, maintenance, license, citations, complete features,
   algorithms, I/O, plots, diagnostics, platforms, dependencies, claims, and
   legally reproducible fixtures.
4. Map each feature to `native`, `equivalent`, `adapter`, `planned`,
   `excluded`, or `not-reproducible`, with linked evidence.
5. Implement justified capabilities locally; keep inference engines optional.

## Exclusion and compatibility policy

Exclusions require the exact feature, evidence, reason, closest workflow, user
impact, and review date. Do not copy incompatible source or trademarked APIs.
Migration adapters are additive and optional.

## Acceptance criteria

Every included feature has executable native/equivalent evidence or a reviewed
disposition. Stable VOIAGE workflows pass with competitor packages absent.

## Out of scope

Unverifiable parity with proprietary or web-only implementation details.

