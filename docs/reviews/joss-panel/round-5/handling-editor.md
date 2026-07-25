# Round 5 handling-editor report

Date: 24 July 2026
Reviewed head: `07cf3f83098952439f7b94773c62a8cbf9c176c1`
Score: 709/1000
Disposition: return for pre-review correction

This is an independent AI-assisted editorial simulation.

## Assessment

All required JOSS sections exist, the paper fits the venue, and the exact
six-page PDF renders cleanly. The numerical example is careful. The submission
should not yet be assigned because no qualifying research use is documented,
the public software identity is internally inconsistent, and the AI disclosure
needs the author's comprehensive attestation.

Non-author use is a valuable signal rather than a universal independent gate.
The minimum unresolved gate is genuine research use by the developer or
another researcher.

## Deductions

| Area | Maximum | Awarded |
| --- | ---: | ---: |
| Scope, history, and research impact | 170 | 85 |
| Editorial structure and format | 110 | 96 |
| Statement of need and audience | 130 | 102 |
| Scholarship and related work | 130 | 98 |
| Software design and claims | 140 | 105 |
| Documentation, tests, and reproduction | 150 | 108 |
| Metadata, authorship, and AI disclosure | 120 | 72 |
| Sentence quality and rendering | 50 | 43 |

## Blocking findings

1. Document completed research use rather than a possible application.
2. Align tag, release, PyPI package, metadata, archive, and paper to one
   revision.
3. Obtain the author's explicit AI-policy affirmation.
4. Correct reviewer-facing documentation errors and broken repository links.

## Editorial and sentence-level findings

- Anchor the need in a realistic mixed-language workflow.
- Do not imply that the common scalar APIs retain the full decision envelope.
- Rewrite the build-versus-contribute paragraph without arguing from an
  unsupported absence.
- State explicitly that only EVPI currently has Python/R/Julia numerical
  parity.
- Shorten estimator mechanics and connect design choices to research meaning.
- Report study costs and EVSI with their value-unit scale.
- Make the sensitivity table directly accessible.
