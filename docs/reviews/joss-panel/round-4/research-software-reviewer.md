# Round 4 research-software review

Date: 24 July 2026  
Score: 927/1000  
Recommendation: major revision

This is an internal AI-assisted readiness simulation, not a formal JOSS review.

## Assessment

`voiage` is research software rather than a one-off script. It has a substantial
test system, documented architecture, an OSI-approved licence, contribution and
support routes, clean Python installation evidence, binding tests, numerical
fixtures, and a coherent worked example. Scientific and numerical accuracy
received full marks in this review.

## Submission blockers

- JOSS requires demonstrated research use, at minimum by the developers; the
  current impact section documents interoperability rather than use.
- No attributable non-author engagement was found.
- Public v1.0.0 lacks the v2 functionality described by the paper.
- The current source, public release, archive, and supplied PDF do not identify
  one immutable revision.
- The AI disclosure does not yet satisfy the tool-version and comprehensive
  human-verification requirements.

## Focused recommendations

- Publish, install-test, archive, and cite v2.0.0.
- State that R and Julia currently require a platform-specific native library.
- Scope field comparisons to the inspected documentation.
- Link the sensitivity CSV directly.
- Rebuild the JOSS PDF from the released commit and inspect every page.
