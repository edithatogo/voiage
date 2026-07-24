# Fail-closed JOSS manuscript rubric

## Purpose

This rubric converts the current JOSS submission, paper-format, review, and
editorial criteria into a repeatable internal review. It is intentionally more
stringent than a normal editorial checklist. A score is evidence about the
quality of the repository's JOSS manuscript package; it is not an acceptance
prediction or an editorial decision.

Authoritative criteria:

- <https://joss.readthedocs.io/en/latest/submitting.html>
- <https://joss.readthedocs.io/en/latest/paper.html>
- <https://joss.readthedocs.io/en/latest/review_criteria.html>
- <https://joss.readthedocs.io/en/latest/editing.html>

## Scoring

| Dimension | Points | Required evidence |
| --- | ---: | --- |
| Scope, significance, and research use | 180 | Research-software fit; concrete, non-aspirational research use; proportionate significance; clear separation of manuscript quality from external adoption. |
| Statement of need and audience | 120 | Specific decision problem, users, consequences of the gap, and relation to existing work, understandable to non-specialists. |
| State of the field and build-versus-contribute case | 130 | Fair comparison with commonly used alternatives; accurate differences, limitations, and a scholarly reason for a separate package. |
| Scientific and numerical accuracy | 150 | Correct VOI concepts, methods, examples, terminology, and bounded claims; no mismatch with source, tests, or release evidence. |
| Software design and research relevance | 100 | Meaningful architectural choices and trade-offs explained in terms of research use, without becoming API documentation. |
| Reproducibility, packaging, documentation, and tests | 100 | Reviewer-installable package; objective functionality checks; documentation, release, archive, and evidence claims trace to repository artefacts. |
| Research-impact statement | 80 | Specific realised impact or credible near-term significance; developer use and independent adoption are not conflated. |
| Structure, metadata, and JOSS format | 60 | Required metadata and sections, 750–1,750 words, appropriate references, valid rendering, and no prohibited full-length-paper or API content. |
| Clarity, accessibility, and sentence quality | 55 | Logical paragraphing, plain language, defined terms, restrained tone, consistent spelling, and sentence-level economy and precision. |
| Citations, provenance, declarations, and AI disclosure | 25 | Claims are supported by authoritative sources; records resolve; archive link, funding, conflicts, authorship, and AI use are complete and accurate. |
| **Total** | **1,000** | |

Each reviewer must:

1. score every applicable dimension and explain every deduction;
2. inspect the whole submission before section, paragraph, claim, citation, and
   sentence-level findings;
3. identify exact line numbers and replacement text where feasible;
4. distinguish manuscript defects from repository defects and external gates;
5. record uncertainty instead of awarding unverified credit;
6. avoid changing scientific meaning merely to improve prose.

## Fail-closed conditions

Any of the following is a manuscript blocker and caps that review at 950:

- a required JOSS section is missing or merely nominal;
- a material scientific, numerical, architectural, release, or impact claim is
  false, unsupported, or contradicted by repository evidence;
- the paper implies independent adoption that has not occurred;
- the state-of-the-field comparison omits a directly relevant common tool or
  lacks a defensible build-versus-contribute rationale;
- an unresolved citation, bibliography, metadata, archive, authorship,
  conflict, funding, or AI-disclosure defect remains;
- the official toolchain cannot render the paper;
- the manuscript contains substantive API documentation or falls outside the
  official word limit;
- sentence-level review is incomplete.

A repository or external-gate finding does not automatically cap the
manuscript score when the paper states the boundary accurately. It is recorded
in a separate readiness assessment and must not be hidden by the score.

## Passing rule

The requested threshold is met only when:

- every full-panel reviewer assigns at least **996/1000** to the complete
  revised manuscript;
- no reviewer records a manuscript blocker;
- all deductions are reconciled in the synthesis;
- the official JOSS build, repository validator, citation audit, claim audit,
  prose checks, and visual PDF review pass at the reviewed revision; and
- a final sentence inventory records no unresolved sentence-level finding.

The panel result is the lowest complete reviewer score, not the mean. No score
is rounded up. Automated agents provide simulated editorial scrutiny; they do
not impersonate JOSS editors or reviewers and cannot make acceptance decisions.

## Submission-eligibility assessment

The final report separately marks each current JOSS pre-review gate as ready,
not ready, or externally pending. In particular, a >995 manuscript result does
not prove demonstrated research impact. The panel must inspect the documented
same-author research integration, while issue #471 seeks non-author activity
as the strong positive signal identified by JOSS rather than treating it as a
universal hard gate.
