# Round 4 panel synthesis

Date: 24 July 2026  
Panel status: major revision  
Passing rule: every role at least 996/1000 and no material blocker

This is an internal AI-assisted readiness panel. It is not a formal JOSS review
and does not predict editorial acceptance.

## Independent scores

| Role | Score |
| --- | ---: |
| Editor-in-chief screening | 761 |
| Handling editor | 834 |
| Domain and health economics | 870 |
| Research software | 927 |
| Reproducibility and packaging | 874 |
| Numerical verification | 932 |
| Accessibility and plain language | 824 |
| Sentence editor | 868 |

No role met the project’s deliberately stringent 996 threshold.

## Consensus

The scientific redesign is successful. The normal--normal study model,
reported results, units, timing, bootstrap interpretation, sensitivity
scenarios, and Rust implementation were independently reproduced. Further
methodological redesign of the worked example is not indicated.

Three external or sequential evidence gates dominate every score:

1. publish and archive the exact v2 software described by the paper;
2. document genuine research use and attributable community engagement;
3. complete the AI disclosure after the author's explicit attestation.

The stale PDF, standalone Julia fixture, changed-line coverage failure, and
generated-artifact drift are repository-owned defects and are being corrected
before the next panel round.

## Implemented from round 4

- Replaced intervention “uptake” language with reduced-form value realisation.
- Added the direct implementation-adjusted EVSI reference.
- Clarified bootstrap scope, study timing, costs, units, and sample-size model.
- Added independent analytical EVPI, EVPPI, and EVSI reference tests.
- Separated the worked example from research impact.
- Bounded comparisons to reviewed tools.
- Simplified the Summary, Statement of need, software-design prose, figure
  caption, and labels.
- Named the sensitivity CSV and recorded deterministic seeds.
- Regenerated the figure and machine-readable outputs.

## Ordered next actions

1. Complete the standalone Julia package test fix.
2. Restore changed-line and changed-branch coverage.
3. Commit, rebase, push, and obtain a fully green hosted revision.
4. Build and inspect the official JOSS PDF from that revision.
5. Merge and publish v2.0.0 with release evidence.
6. Request and verify the v2 Software Heritage snapshot.
7. Replace v1 and prospective availability text with immutable v2 facts.
8. Resolve the research-use and AI-attestation gates.
9. Run a fresh independent panel against the exact released revision.
