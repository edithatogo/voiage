# Track Specification: R Package Documentation Manual And Vignette

## Overview

The `r-package/voiageR` binding currently ships roxygen-based Rd help pages and package tests, but it does not yet provide a publication-quality long-form documentation path. This track will bring the R package up to a release-grade documentation standard by adding:

- a clean package-level help topic
- a concise getting-started vignette or equivalent long-form guide
- a reproducible PDF reference manual generated from the R package documentation
- the CI/release checks and contributor guidance needed to keep those artifacts current

The target is an R package documentation surface that is suitable for source distribution, release artifacts, and future CRAN-style maturation.

## Goals

1. Provide a package-level documentation entry point for `voiageR`.
2. Remove stale or misleading documentation metadata in the R binding.
3. Add narrative documentation that shows the intended R workflow end to end.
4. Produce a latex-style PDF manual artifact from the package documentation.
5. Make the documentation build path explicit in CI, release automation, and contributor guidance.

## Functional Requirements

1. The R package must expose a generated package-level help topic that matches the current exported surface.
2. The roxygen comments, generated Rd files, and package metadata must be kept in sync.
3. The package documentation must include at least one long-form narrative guide that demonstrates:
   - package setup and Python environment selection
   - a minimal EVPI, EVPPI, or EVSI workflow
   - the intended error-handling or environment caveats for the binding
4. The package must produce a PDF manual artifact from its documentation sources.
5. The manual and any vignette sources must build deterministically in non-interactive CI conditions.
6. The contributor and release docs must explain:
   - how to build the R package
   - how to generate or verify the PDF manual
   - whether vignette generation is part of the release path

## Non-Functional Requirements

1. Keep the documentation workflow compatible with standard R tooling.
2. Avoid introducing heavy documentation infrastructure unless it clearly improves the package’s release-quality docs story.
3. Keep the build path reproducible and suitable for automation.
4. Do not change the Python-facing runtime behavior unless the doc-cleanup work reveals a concrete contract bug.

## Acceptance Criteria

1. The package ships a generated package-level help topic and current Rd pages for the public R API.
2. A narrative R documentation source exists and renders cleanly under CI.
3. A PDF manual can be built from the package documentation without manual intervention.
4. The release and contributor docs state how the manual is produced and validated.
5. The package still passes its build and check gates after the documentation changes.
6. There are no stale roxygen tags or obvious documentation drift in the R package source.

## Out Of Scope

1. Reworking the Python core API.
2. Adding pkgdown unless it becomes necessary to deliver the manual/vignette goals.
3. Changing the package’s publication channel beyond documentation artifact support.
4. Major R API redesigns unrelated to documentation quality.

## Execution Notes

- Prefer small, parallelizable slices so separate subagents can own package metadata, narrative docs, manual generation, and CI/release updates independently.
- Treat the PDF manual as a first-class release artifact, not an afterthought.
- Keep the documentation policy explicit: if a vignette is included, it must be validated and documented; if the package remains reference-manual only, that decision must be written down.
