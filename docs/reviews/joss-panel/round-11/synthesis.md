# Round 11 synthesis

## Reviewed evidence

The panel assessed commit `6b16897bcf51d2d78233accc190e47f4aaef4d0e` and its
successful official JOSS workflow `30252163688`. The rendered PDF had six
pages and was visually inspected. Automated contract, SourceRight, Authentext,
and deterministic reproduction checks passed for the reviewed source.

## Accepted corrections

1. Replace `Figure \autoref{...}` with `\autoref{...}` to avoid “Figure Figure
   1” in the PDF.
2. Define the practical meaning of EVPI, EVPPI, EVSI, and ENBS in the summary.
3. Explain the separate core through analyst choice and shared decision context,
   rather than software-boundary vocabulary alone.
4. Explain the population meaning of the per-person EVPI result.
5. State that the sensitivity analysis changes the assumed individual
   study-outcome standard deviation.

## Residual gate ledger

| Gate | Status | Owner |
| --- | --- | --- |
| Exact-source contract, SourceRight, Authentext, and regeneration | Re-run required after accepted corrections | Repository |
| Exact-source hosted JOSS render and visual review | Re-run required after accepted corrections | Hosted workflow |
| Final claim-to-source verification | Pending | Human author |
| Complete AI-output review/validation attestation | Pending | Human author |
| Completed research workflow using released software | Pending | External evidence |
| Non-author engagement or collaborative input | Pending | External evidence |

The panel scores cannot legitimately reach >995/1000 until the human evidence
is available. The next round must review the exact post-correction hosted
artifact and maintain that boundary.
