# Track Specification: Binding Registry Live Verification

## Overview

The project has automation to prepare and publish release artifacts, but it still
needs a maintained, reviewable proof of live registry state for each language
binding after each intended release.

This track creates a lightweight verification contract that captures where each
binding is currently confirmed live, pending, or external/manual, and where action
is required.

## Functional Requirements

1. Define a stable per-language registry evidence schema that records package,
   registry endpoint, check date, outcome, and evidence link.
2. Document a repeatable manual/automated audit workflow for:
   - Python (`voiage` on PyPI and conda-forge PR status),
   - R (`voiageR` on CRAN and r-universe),
   - Julia (`Voiage` on Julia General),
   - TypeScript (`@voiage/core` on npm),
   - Go (`github.com/edithatogo/voiage/bindings/go` on module proxy),
   - Rust (`voiage-core` on crates.io),
   - .NET (`Voiage.Core` on NuGet).
3. Update `docs/release/binding-submission-checklist.md` and `docs/release/polyglot-bindings.md`
   to separate automated submission steps from external registry confirmation.
4. Keep CPU-only proof artifacts reproducible; do not assume any external action
   has succeeded without explicit evidence.

## Non-Functional Requirements

1. No runtime code changes; this is a metadata, doc, and process track.
2. Preserve historical audit entries so regressions are visible after future release cycles.
3. Avoid over-claiming; "submitted" remains a boundary term unless registry evidence is present.

## Acceptance Criteria

1. The track defines a durable and versioned registry-audit evidence format.
2. At least one registry evidence artifact exists per binding language.
3. The existing checklist and matrix reflect live confirmation status per binding.
4. A clear owner/next-step is recorded for each package that remains unconfirmed.

## Out of Scope

1. Performing external registry account operations from this repository.
2. Adjusting publication workflows or credential flows.
3. Any change to runtime API behavior or numerical methods.
