# Round 5 editor-in-chief screening

Date: 24 July 2026
Reviewed head: `07cf3f83098952439f7b94773c62a8cbf9c176c1`
Score: 705/1000
Disposition: do not submit; likely desk rejection in the reviewed state

This is an independent AI-assisted screening simulation, not a JOSS editorial
decision.

## Assessment

The software is substantial, within JOSS scope, and supported by unusually
strong testing and documentation infrastructure. The current submission still
fails the demonstrated-research-use screening gate. The synthetic example
proves functionality, while the same-author `vop_poc_nz` bundle proves an
interoperability contract; neither records use of `voiage` to answer a research
question.

The reviewed branch also lacked one coherent public identity. The paper
described v2 functionality, public package metadata named v2.0.0, and the only
public release and Software Heritage snapshot covered v1.0.0.

## Deductions

| Criterion | Maximum | Awarded |
| --- | ---: | ---: |
| Scope fit and scholarly significance | 200 | 155 |
| Substantial effort and open development | 180 | 165 |
| Research use and community evidence | 160 | 20 |
| Paper compliance and editorial quality | 200 | 170 |
| Functionality, documentation, and assurance | 160 | 150 |
| Release, reproduction, and archival identity | 100 | 45 |

## Blocking findings

1. Record actual developer research use, with a research question, attributable
   inputs, commands, outputs, interpretation, and immutable software identity.
2. Publish and archive the exact reviewed release.
3. Complete the author-confirmed AI-policy attestation.
4. Replace prospective availability text with observed release evidence.

## Sentence-level findings

- Remove “Implementation” from the title because implementation-adjusted EVSI
  is outside the demonstrated stable scope.
- Qualify the claim that calculations retain context: the provenance-bearing
  envelope is an opt-in Python interface.
- State that R and Julia demonstrate shared scalar EVPI, not semantic-envelope
  parity.
- Reduce the paper below the formal word-limit boundary.
- Replace the future-tense availability paragraph after the exact release
  exists.
