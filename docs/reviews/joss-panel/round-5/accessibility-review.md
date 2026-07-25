# Round 5 accessibility and plain-language review

Date: 24 July 2026
Reviewed head: `07cf3f83098952439f7b94773c62a8cbf9c176c1` plus the then-current
working-tree manuscript
Score: 583/1000
Disposition: substantial revision

This was an independent, read-only, AI-assisted accessibility review. It is an
internal editorial diagnostic, not a JOSS decision. The reviewer examined the
complete manuscript, the rendered figure, the earlier six-page workflow PDF,
the public documentation, and the current JOSS criteria.

## Deductions

| Area | Deduction |
| --- | ---: |
| Title and positioning | 18 |
| Summary and statement of need | 74 |
| Jargon and abbreviations | 95 |
| Structure and cognitive load | 47 |
| Decision-science value versus engineering | 62 |
| Worked example and figure | 38 |
| PDF and digital accessibility | 60 |
| Readiness and evidence clarity | 23 |

## Principal findings

- The central research message was displaced by software architecture,
  validation vocabulary, release mechanics, and several abbreviations in the
  opening paragraph.
- The summary did not yet explain the software to applied researchers in
  healthcare, policy, economics, business, or related fields.
- The design section used implementation terms where the reader needed to know
  why the architecture protects the meaning of a decision.
- The displayed analytical equation was scientifically sound but too dense for
  the main JOSS narrative; the full model should remain available in
  reproducibility material.
- The worked example should lead with the decision and its consequences, then
  give technical assumptions and simulation diagnostics.
- Figure labels should explain the research questions, expand or avoid
  abbreviations, show bar values, and have a directly accessible tabular
  equivalent.
- The previous workflow PDF could not be approved as the final rendering
  because it predated the revised manuscript. It was also untagged, as is
  typical of the JOSS template, so the source paper should provide a textual
  data equivalent rather than describe its image caption as tagged
  alternative text.

## Sentence-level priorities

1. Introduce the four VOI measures as questions before their abbreviations.
2. Replace “contract”, “kernel”, “fixture”, “parity”, and “envelope” where
   ordinary language carries the same meaning.
3. State plainly that Python has the broad workflow while R and Julia currently
   share only EVPI.
4. Explain net benefit before reporting the health example.
5. Present the substantive EVPI, EVPPI, EVSI, and ENBS findings before seeds,
   interval methods, and theoretical checks.
6. Replace “reduced-form realisation assumption” with a plain explanation of
   how much evidence changes practice.
7. Recast the same-author integration as workflow evidence without calling it
   independent adoption.

## Recommendations accepted in the subsequent revision

- The title, summary, statement of need, field comparison, software-design
  rationale, worked example, and impact statement were rewritten for a
  non-specialist reader.
- The complete analytical equations and benchmarks were moved to
  `paper/health-example-methods.md`; the paper retains the assumptions and
  scientific rationale.
- The figure now asks plain-language questions, expands the plotted measures,
  labels the bars, and links to the underlying tables.
- Prospective release and archival claims remain explicit rather than being
  rewritten as completed evidence.

## Recommendations not treated as established facts

The reviewer described non-author use as a stand-alone research-impact blocker.
The official criteria are more specific: credible near-term significance may
satisfy the impact section, but a single-author project with no community
engagement, external use, or collaborative input is separately classified as
not acceptable. The readiness record therefore tracks both the reproducible
near-term-significance evidence and the unresolved community-engagement gate.
