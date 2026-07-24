# Round 4 handling-editor review

Date: 24 July 2026  
Score: 834/1000  
Recommendation: major revision

This is an internal AI-assisted readiness simulation, not a JOSS editorial
decision.

## Whole-submission assessment

The paper has a clear statement of need, a restrained comparison with existing
tools, an explicit Rust/Python/R/Julia boundary, and a reproducible health
example. The revised normal--normal EVSI model resolves the main scientific
weakness in the earlier draft. The remaining problems concern submission
identity and evidence rather than a further methodological redesign.

## Priority findings

1. Publish one exact v2 release containing the reviewed functionality and bind
   the paper, release, evidence manifest, and archive to the same commit.
2. Replace the research-impact section only when a genuine research execution
   or attributable external engagement can be cited.
3. Complete the JOSS AI disclosure only after the author confirms the required
   human decision, review, modification, and validation statements.
4. Rebuild and visually inspect the JOSS PDF after the source is frozen.
5. Keep the worked example separate from the research-impact statement.
6. Describe the 60% scalar as value realisation, not intervention uptake or
   implementation-adjusted EVSI.

## Section and sentence findings

- Bound comparisons to the tools and documentation reviewed.
- Use “main design boundary” rather than “primary boundary”.
- State units for both EVPPI estimates.
- Describe bootstrap intervals as Monte Carlo uncertainty over paired PSA
  draws.
- State the benefit and cost timing assumptions.
- Replace “delivery assumptions” with the exact timing and value-realisation
  assumptions.
- Name the machine-readable sensitivity file.
