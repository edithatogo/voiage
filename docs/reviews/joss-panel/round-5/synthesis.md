# Round 5 panel synthesis and revision disposition

Date: 24 July 2026
Review target: `paper.md`, the repository package, and the available workflow
PDF at head `07cf3f83098952439f7b94773c62a8cbf9c176c1`

This synthesis combines independent AI-assisted editorial simulations. It does
not predict or replace a human JOSS decision. Scores are internal diagnostics;
the lowest substantiated finding, rather than an average, determines the next
revision.

## Separate reports

| Role | Score | Disposition |
| --- | ---: | --- |
| Editor-in-chief screening | 705 | Do not submit |
| Handling editor | 709 | Return for pre-review correction |
| Domain and health-economics reviewer | 747 | Major revision |
| Research-software reviewer | 673 | Major revision |
| Reproducibility and packaging reviewer | 754 | Major revision |
| Sentence editor | 815 | Major revision |
| Accessibility and plain-language reviewer | 583 | Substantial revision |

## Agreement across the panel

Every role agreed that the software is within JOSS's subject matter and that the
health-example calculations are useful. The panel also agreed on four
submission-level defects:

1. the public v1 release does not contain the v2 functionality described by the
   manuscript;
2. the paper and public interfaces previously blurred Python's optional
   provenance-bearing record with the scalar R and Julia EVPI bindings;
3. a single-author project needs genuine human community engagement, external
   use, or collaborative input under the current screening criteria; and
4. the final AI disclosure requires an author-confirmed account of human design,
   review, editing, and validation, including the best available tool-version
   record.

The sentence and accessibility reviewers additionally agreed that the paper
gave implementation mechanics more attention than the research decision. The
domain reviewer required the complete analytical model and independently
checked benchmarks to remain available.

## Resolved conflicts

### Plain language versus scientific completeness

The accessibility reviewer recommended removing the displayed analytical
equation; the domain reviewer required the full model and preposterior
variance. The revision keeps the assumptions and purpose in the JOSS paper and
moves the complete equations, definitions, and benchmarks to
`paper/health-example-methods.md`. This keeps the scientific record inspectable
without making the main narrative depend on specialist notation.

### Research impact versus independent engagement

Some reports treated non-author research use as the only impact route. The
official 2026 criteria permit specific reproducible materials to demonstrate
credible near-term significance. They separately classify a single-author
project with no community engagement, external use, or collaborative input as
not acceptable. The revision therefore records reproducible near-term
significance without claiming that it resolves the human engagement gate.

### Architecture detail versus JOSS's design requirement

JOSS excludes API documentation but requires design trade-offs and their
research implications. The revision removes interface inventory and preserves
one rationale: implementing selected calculations once in Rust reduces
cross-language drift, while requiring native builds and limiting the current R
and Julia surfaces.

## Prioritised changes

### Submission-level

- Publish, test, and archive the exact v2 release described by the paper.
- Replace v1 and prospective availability wording with observed v2 evidence.
- Obtain attributable human engagement, external use, or collaborative input;
  do not count agents, bots, or the author's repositories.
- Obtain the author's explicit AI-policy attestation and record all tool
  versions that can be established without guessing.
- Rebuild and visually inspect the Open Journals PDF from the final revision.

### Manuscript-wide

- Lead with the research decision and the value of reducing uncertainty.
- Explain the four measures as questions before introducing abbreviations.
- State the current language boundary accurately: broad Python workflow and
  shared scalar EVPI in R and Julia.
- Compare related software by research purpose and depth rather than by a
  feature catalogue.
- Explain why the Rust-centred design matters and state its trade-offs.

### Section and paragraph

- Put the substantive health-example findings before simulation diagnostics.
- Explain net benefit and the sign change in ENBS in ordinary language.
- Keep the proposed study explicitly synthetic and non-prescriptive.
- Link the figure to machine-readable tables and the full analytical method.
- Keep the same-author integration separate from independent adoption.

### Sentence and figure

- Replace opaque terms such as “result envelope”, “kernel”, “fixture”, and
  “reduced-form realisation” where plain wording is exact.
- Use restrained verbs and avoid claims of shared semantic parity.
- Ask plain-language questions in the figure, expand plotted measures, and label
  bar values.
- Keep units, timing, evaluated sample-size brackets, and uncertainty scope
  explicit.

## Implemented after the reports

The working-tree revision:

- changed the title to focus on value-of-information analysis for research
  decisions;
- rewrote the summary, statement of need, state of the field, design section,
  worked example, and impact statement;
- narrowed all cross-language claims to the executable interfaces;
- moved the full study equations and independently checked benchmarks to a
  dedicated reproduction note;
- added a structured reproduction manifest and clean-regeneration verification;
- corrected public documentation and repository links;
- deleted the stale supplementary note that called a plug-in EVPI estimator
  unbiased and described unsupported methods; and
- revised and visually inspected the health figure, including its tabular text
  equivalents.

## Exit criteria for the next loop

The next panel must review a newly built PDF and the exact current source. No
role may score above 995/1000 while a factual, policy, release, accessibility,
or evidence blocker remains. Scores from this round are superseded only by a
complete fresh review; they are not carried forward or adjusted arithmetically.
