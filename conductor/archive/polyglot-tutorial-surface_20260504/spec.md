# Track Specification: Polyglot Tutorial Surface And Worked Examples

## Overview

`voiage` now has a strong Python notebook surface, but the tutorial story is uneven across the binding ecosystem. This track will make the “how do I use this in practice?” layer explicit and consistent across the library’s supported languages, with the right artifact type for each ecosystem:

- Python: runnable Jupyter notebooks and example guides
- R: vignette-style long-form documentation and package manual coverage
- Julia, Go, Rust, TypeScript, .NET: practical code snippets, runnable sample programs, and README-level walkthroughs

The goal is not to force every language into the same format. The goal is to give each language a clear, discoverable, publication-quality tutorial surface that demonstrates the same key VOI use cases and maps back to the same core library concepts.

## Goals

1. Make the main user journeys easy to discover in every supported language.
2. Keep tutorial artifacts aligned with the actual package APIs and release paths.
3. Demonstrate the canonical VOI workflows in a language-appropriate way.
4. Make tutorial sources executable or at least smoke-testable where the ecosystem allows it.
5. Keep the tutorial surface connected to the main docs and release guidance.

## Functional Requirements

1. The Python surface must remain a first-class notebook/tutorial experience with clear entry points for:
   - getting started
   - EVPI / EVPPI / EVSI
   - network meta-analysis and structural VOI
   - advanced or cross-domain use cases
2. The R surface must provide a publication-quality tutorial path, including:
   - a narrative vignette or equivalent long-form guide
   - package-level examples that show EVPI / EVPPI / EVSI usage
   - documentation that explains environment setup and the R-to-Python bridge
3. Each non-Python binding must include at least one tutorial-style example that shows:
   - setup or import
   - a small real-looking VOI workflow
   - how to run or verify the example
4. The tutorial surface must be discoverable from the top-level docs and from each language binding’s own README or guide.
5. Tutorial sources should be runnable or smoke-testable whenever the language ecosystem makes that practical.
6. The tutorial set should cover the key use cases for:
   - basic VOI
   - sample information / partial perfect information
   - domain-specific workflow examples
   - advanced methods where they are already stable

## Non-Functional Requirements

1. Keep the tutorial formats native to each language ecosystem.
2. Prefer small, deterministic examples over large synthetic benchmarks.
3. Avoid coupling tutorial materials to internal implementation details.
4. Keep the tutorial surface stable enough to support release documentation and external adoption.

## Acceptance Criteria

1. Python has a curated notebook/tutorial index that surfaces the main workflows.
2. R has a long-form tutorial artifact and documentation guidance suitable for release distribution.
3. Julia, Go, Rust, TypeScript, and .NET each have a practical tutorial or walkthrough in the repo.
4. The top-level documentation links users to the right tutorial entry points for each language.
5. Tutorial artifacts are validated by tests, build checks, or smoke checks appropriate to each language.
6. The docs clearly distinguish notebook tutorials, package examples, and release-quality walkthroughs.

## Out Of Scope

1. Rewriting the core Python APIs.
2. Mandating Jupyter notebooks for languages where they are not natural.
3. Adding brand-new bindings.
4. Changing package publishing channels purely to support tutorials.

## Execution Notes

- Keep the work split by language family so separate subagents can own Python, R, and the other binding families without overlapping writes.
- Use the smallest artifact that gives each ecosystem a credible tutorial story.
- Where an ecosystem already has README examples, decide whether that is sufficient or whether it needs a more structured tutorial page before adding heavier machinery.
