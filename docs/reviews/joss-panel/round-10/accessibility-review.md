# Accessibility and applied-research review

## Role and basis

This independent panel role assessed the complete initial Round-10 manuscript,
its claim ledger, assurance records, fixed-seed health example, and rendered
figure. It inspected the article from an applied reader's perspective,
including policy, healthcare, economics, business, and marketing readers who
do not necessarily work in software development or value-of-information (VOI)
methods.

## Initial assessment

Score: **872/1000**. The score was capped because the source-bound editorial
assurance was stale and the manuscript asserted a final human AI-review
attestation that was still pending in its assurance record. The article was
scientifically careful, but technical vocabulary and the original caption made
the health example less independently interpretable than it should be.

## Findings and required changes

1. Define EVPI, EVPPI, EVSI, and ENBS as the value of improving a decision,
   rather than as compressed technical labels.
2. Explain why preserved decision context matters to an applied analyst and
   give bounded non-health examples without claiming non-health validation.
3. Translate the Rust-core design into consequences for consistent results,
   language choice, and installation burden.
4. State the normal study model in plain language, including equal allocation,
   known outcome variability, and the health-gain assumption.
5. Distinguish finite-simulation error from uncertainty about the model itself.
6. Give the principal result in each figure panel and make the EVPPI bars
   distinguishable without colour alone.
7. Keep the research-impact boundary candid: the same-author exchange is a
   verification case, not independent adoption or completed research use.

## Revision disposition

All seven changes were accepted in the Round-10 revision. The revised figure
uses a hatch and outline for programme cost as redundant visual encoding. The
remaining research-use and non-author-engagement findings are external JOSS
eligibility gates, not defects that prose can repair.

## Re-review criteria

The next panel round must inspect the exact committed source and PDF, verify the
new figure visually, and confirm that all specialised terms have a nearby plain
language explanation. It must also separately recheck the human-only AI and
citation attestations.
