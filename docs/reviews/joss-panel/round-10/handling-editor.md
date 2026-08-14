# Handling-editor review

Score: **854/1000**
Disposition: **major pre-review revision**
Fail-closed cap: **applies**

## Article architecture

The article moves coherently from the research decision to need, field
position, design, example, impact, declarations, and availability. All required
JOSS sections are substantive. The worked example is near the upper
proportional limit but remains relevant and does not become API documentation.

## Manuscript blockers

1. The Julia platform sentence was stale against the current binding workflow.
2. The build-versus-contribute case did not squarely answer why an existing
   environment-specific package was insufficient.
3. Citation and AI human-assurance states were incomplete.

## Recommended revisions

- Explain that a language-neutral core avoids making one specialist runtime and
  data model authoritative for the other languages.
- Tie stability requirements to each interface that actually exposes a method.
- Use paired-bootstrap Monte Carlo terminology for every reported interval.
- Describe ENBS crossings only among evaluated sample sizes.
- Remove subjective words such as “thoroughly”.
- Identify the exact release commit in the availability section.

## Readiness boundary

The article can be improved locally, but actual research use and human
attestations cannot be created through prose.
