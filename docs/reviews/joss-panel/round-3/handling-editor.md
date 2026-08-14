# Round 3 independent JOSS handling-editor report

## Editorial status

**Recommendation:** major revision; do not submit the current revision.

**Score:** **821/1000**.

**Fail-closed status:** manuscript and submission-package blockers are present.
The score is capped below 950 until the paper, software, tests, release, archive,
and AI disclosure describe one immutable reviewed revision.

This is an internal, AI-assisted editorial simulation, not a decision or review
issued by the Journal of Open Source Software (JOSS). JOSS's current policy
reserves evaluative editorial and reviewer judgements for humans. The report can
inform author revision, but it must not be represented as an independent human
review or submitted as a JOSS reviewer report.

## Revision reviewed

- Canonical JOSS source: `paper.md`.
- Committed base: `1dfcd3d42af591ca15fcc03ad958123cc153dbbf`.
- Additional state: the uncommitted working tree on 24 July 2026.
- Public release cited by the paper: `v1.0.0`, commit
  `05cc373d78ae74143194e889ff1317de4dfea52e`.
- Separate arXiv source: not reviewed.

The working-tree distinction is material. `paper.md`, the revised EVSI
implementation, the v2 contract, binding manifests, and supporting tests differ
from the committed base. The latest successful hosted JOSS build is for
`1dfcd3d42af591ca15fcc03ad958123cc153dbbf`, not for the manuscript assessed
here.

## Criteria applied

This assessment uses the current:

- [JOSS submission and pre-review criteria](https://joss.readthedocs.io/en/latest/submitting.html);
- [JOSS paper requirements](https://joss.readthedocs.io/en/latest/paper.html);
- [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html);
- [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html);
- [JOSS AI-use policy](https://joss.readthedocs.io/en/latest/policies.html).

The local fail-closed rubric at `docs/reviews/joss-panel/rubric.md` was also
applied. Authentext was used for claim and prose hygiene. SourceRight's
provenance command was run read-only, but its present parser did not recognise
the Pandoc citation syntax or BibTeX records and returned no sources. Its
unsupported-claim list is therefore not treated as a substantive citation
verdict.

## Executive assessment

The manuscript has a credible JOSS shape. It is short, restrained, and mostly
accessible. It explains VOI through research questions, names the intended
users, compares the software fairly with relevant R, web, and Python tools,
states architectural trade-offs, and reports a reproducible synthetic health
example. It does not disguise the narrow R and Julia interfaces or claim
independent adoption that has not occurred.

The submission case is nevertheless not ready:

1. The cited `v1.0.0` release does not contain the analytical EVSI
   implementation or revised generic EVSI contract described in the paper.
2. The current manuscript and implementation are uncommitted, unreleased, and
   not covered by a hosted Open Journals build at the same revision.
3. The availability section promises a future release instead of identifying
   the exact software being reviewed.
4. The AI disclosure does not yet use JOSS's required assurance that the human
   author reviewed, modified, and validated all AI-assisted outputs and made
   the primary design decisions. It also gives families or service-managed
   models rather than every recoverable model version.
5. The research-impact case is specific but weak. The synthetic example is a
   paper demonstration, and the cited same-author `vop_poc_nz` artefact is an
   interoperability contract bundle rather than clear evidence that the
   released `voiage` package was used to complete a research analysis.
6. No attributable non-author installation, use, or collaborative input is
   recorded. Current JOSS submission guidance does not make external
   contribution universally mandatory for a well-maintained solo project, but
   its reviewer guidance calls a single-author project with no community
   engagement, external use, or collaborative input unacceptable. This is a
   substantial screening risk, not a wording problem that can be edited away.

The scientific worktree has improved since the preceding sentence report.
Focused tests now support the manuscript's fitted joint-Gaussian description:
the current and posterior decisions and prior-predictive data use the same
fitted prior, Gaussian draws retain their tails, correlated parameters update
together, and invalid covariance and scale cases fail closed. These passing
local tests do not cure the release-identity defect.

## Pre-review and repository assessment

| Requirement | Assessment | Evidence and editorial finding |
| --- | --- | --- |
| Research-software scope | **Ready** | VOI is an established research method, and the package implements a mathematical decision-analysis library. |
| OSI-approved licence | **Ready** | Root `LICENSE` is Apache-2.0. |
| Public repository and issue route | **Ready** | The GitHub repository is public and permits public issues and proposed changes. |
| More than six months of public development | **Ready** | GitHub records repository creation on 3 July 2025. The local history spans July 2025 to July 2026 and is not a single repository dump. |
| Iterative development | **Ready with concentration noted** | Commits occur across the year, although activity is bursty and heavily concentrated in April and July 2026. |
| Major author contribution | **Ready** | The named author is the dominant human contributor. Automated accounts are not treated as co-authors or community evidence. |
| Open-source practice | **Ready** | Releases, changelog, tests, CI, documentation, contribution guidance, support, and governance routes are present. |
| Reviewer installation | **Historically ready; current revision unproven** | `v1.0.0` is installable, but it is not the software described. The uncommitted v2 candidate has no matching published reviewer installation. |
| Demonstrated developer research use | **Questionable** | The health example demonstrates capability. The `vop_poc_nz` contract is research-adjacent and same-author, but current evidence does not clearly show the released package completing a research analysis. |
| External engagement | **Not evidenced** | Issue #471 contains an author request only. No non-author report, issue, pull request, use, or attributable domain review was found. |
| Feature completeness | **Not established for the described revision** | The paper calls the generic two-loop estimator developing and the compatibility estimators non-stable. That candour is good, but the release and stable scientific surface must be unambiguous at submission. |
| Immutable reviewed release | **Blocked** | `v1.0.0` lacks the described EVSI work; the intended replacement release does not yet exist. |
| Archive | **Historical snapshot ready; reviewed revision blocked** | The SWHID resolves and contains the `v1.0.0` tag, but it does not identify a released version of the present worktree. |
| Official paper build | **Blocked for this revision** | The hosted JOSS workflow passed at the committed base, not at the dirty revision reviewed here. |
| AI disclosure | **Blocked** | Tools and activities are named, but version identification and the required all-output human verification assurance are incomplete. |

## Required paper sections

| Section | Assessment |
| --- | --- |
| Summary | **Present and substantive.** The purpose and four VOI questions are accessible. The language-interface catalogue and normal-prior terminology are more technical than necessary for the non-specialist opening. |
| Statement of need | **Present; revision required.** It identifies the problem, consequences, users, and possible domains. It does not itself explain the relation to existing software or state the exact ecosystem gap, as the current JOSS criterion asks. |
| State of the field | **Present; revision required.** It treats `voi`, `BCEA`, `dampack`, SAVI, and two Python tools fairly. The build-versus-contribute case remains implicit rather than directly explaining why contribution to an existing project would not meet the research requirement. |
| Software design | **Present and mostly substantive.** The shared-kernel trade-off is meaningful and connected to cross-language research use. Some maturity and Monte Carlo detail reads like internal method documentation, and it describes unreleased behaviour. |
| Research impact statement | **Present but weak.** The example is concrete and reproducible, but synthetic and author-created. The second artefact is a compatibility bundle, not clearly a completed analysis using the released software. |
| AI usage disclosure | **Present but policy-incomplete.** Scope and accountability are stated, but exact recoverable versions and JOSS's required human review, modification, and validation assurance are missing. |
| Acknowledgements | **Present and complete on the author's stated facts.** Funding and competing interests are explicit. |
| References | **Present.** Fourteen in-text keys reconcile with fourteen BibTeX records. The software release and Software Heritage snapshot are cited, but they do not identify the software currently described. |
| Software and data availability | **Useful additional section, but blocked.** It is candid about the mismatch while leaving the reviewed version prospective. |

## Score

| Dimension | Score | Maximum | Deduction |
| --- | ---: | ---: | --- |
| Scope, significance, and research use | 130 | 180 | Clear research application and credible potential, but the demonstrated-use evidence is same-author and does not clearly show the released package completing research. No external engagement is evidenced. |
| Statement of need and audience | 111 | 120 | Specific problem and users; relation to existing software and the precise ecosystem gap are deferred to the next section rather than stated here. |
| State of the field and build-versus-contribute case | 117 | 130 | Relevant alternatives are cited and treated fairly. The reason for a separate package is still implicit and partly architectural rather than a direct research-need argument. |
| Scientific and numerical accuracy | 143 | 150 | The current focused scientific tests support the revised Gaussian and analytical EVSI claims. The uptake sentence implies separate causal effects from a scenario that changes delay and uptake together, and no immutable release contains the assessed implementation. |
| Software design and research relevance | 92 | 100 | A real architecture and trade-off are explained. Some text is method-governance detail and some is more technical than the JOSS audience needs. |
| Reproducibility, packaging, documentation, and tests | 58 | 100 | Strong repository infrastructure and passing focused checks, but no committed, hosted, installable, archived revision matches the paper. |
| Research-impact statement | 38 | 80 | Reproducible synthetic material is useful, but realised impact, external use, and actual package use in the cited workflow are not established. |
| Structure, metadata, and JOSS format | 59 | 60 | Required metadata and sections are present; the 1,544-word body is within the 750–1,750 range. The date must be regenerated for the final immutable revision. |
| Clarity, accessibility, and sentence quality | 50 | 55 | Generally clear and non-promotional. Several design and assurance sentences remain technical or abstract. |
| Citations, provenance, declarations, and AI disclosure | 23 | 25 | Citation keys reconcile and declarations are present, but the release/archive citation mismatch and incomplete AI-policy assurance are material. This dimension is not reduced further because those defects are also captured in the reproducibility score. |
| **Total** | **821** | **1000** | **Major revision; do not submit.** |

## Sentence-level changes

The following inventory lists every substantive sentence in the current
`paper.md` that still requires revision. Sentences not listed were reviewed and
do not require an editorial change in this round. A line can remain blocked by
external evidence even when its current wording is honest.

### P0: submission blockers

#### 1. Lines 45–47: unreleased Rust EVSI claim

Current claim:

> Rust provides the shared input rules and selected calculations, including
> EVPI and EVSI for a two-arm study with a normal prior and normal likelihood.

The source worktree supports this statement, but the cited `v1.0.0` release
does not. Retain it only after the exact reviewed release includes the
analytical route. Otherwise limit the Summary to the capabilities in
`v1.0.0`.

Suggested post-release form:

> In version `<reviewed-version>`, Rust provides the shared input checks and the
> EVPI and two-arm analytical EVSI calculations.

Do not insert a version placeholder into the submitted paper. Replace it with
an identifier only after that release exists.

#### 2. Lines 136–141: assurance and release describe different software

Current sentences:

> The current repository tests calculations against known results, rejects
> invalid data, checks repeatability across implementations, and exercises clean
> installations on supported operating systems. Release 1.0.0 contains a source
> distribution, three platform wheels, and checksums. The release record does
> not contain the analytical EVSI implementation or the revised generic EVSI
> contract described here.

The last sentence correctly identifies a submission blocker; it does not solve
it. A JOSS paper should not describe one software state and cite another.
Replace all three sentences after publishing the reviewed revision and after
its hosted checks pass.

Suggested post-release form:

> The tests for version `<reviewed-version>` compare the calculations with
> independent results, reject invalid data, check agreement across the supported
> interfaces, and exercise clean installations on Linux, macOS, and Windows.
> The release provides the reviewer-installable source and wheel artefacts
> listed in its release-evidence manifest.

Name only platforms and artefacts that the immutable hosted record verifies.

#### 3. Lines 193–197: AI systems and versions are not fully identified

Current sentences:

> OpenAI Codex, using GPT-5-family models, and Google Jules, using
> service-managed models, assisted with repository analysis, code and test
> drafting, refactoring, documentation, workflow review, and manuscript editing.
> Exact model identifiers were not retained for every historical session.

JOSS asks for the systems and versions used and where they were applied.
“GPT-5-family” and “service-managed models” are not versions. Recover model
identifiers from available session, task, and service records. Where a service
did not expose a model identifier, say that specifically and identify the
service version or access dates that can be verified.

An exact replacement cannot be supplied until that provenance has been
reconciled. Do not invent historical identifiers.

#### 4. Lines 197–200: human-verification assurance is narrower than JOSS policy

Current sentence:

> The human author selected the research problem and architecture, reviewed this
> manuscript, and validated the reported code, references, and numerical results
> against repository tests and generated evidence.

JOSS currently asks the authors to affirm that humans reviewed, modified, and
validated all AI-assisted outputs and made the primary architectural and design
decisions. The current statement covers the manuscript and reported subset, but
not all AI-assisted code, tests, documentation, and prose; it also omits
“modified”. Replace it only if the author can personally attest to the stronger
statement.

Suggested form, conditional on author confirmation:

> The human author made the primary research, architectural, and design
> decisions and reviewed, modified, and validated all AI-assisted code, tests,
> documentation, references, and manuscript text.

#### 5. Lines 211–218: availability is historical and prospective, not exact

Current sentences:

> The Python package and release 1.0.0 are public. The fixed-seed health-example
> script and its machine-readable outputs use synthetic data. The repository is
> preserved by Software Heritage as [SWHID]. Release 1.0.0 predates the revised
> EVSI contract described here. The submitted paper will cite a release made
> from the exact reviewed revision, together with its release-evidence manifest.

The first two facts are accurate. The SWHID identifies a snapshot that contains
the `v1.0.0` tag, not an archive of the changing worktree. The final two
sentences state that the cited release is wrong and substitute a promise for
availability evidence.

After publishing and archiving the reviewed revision, replace the paragraph
with factual identifiers:

> Version `<reviewed-version>`, commit `<full-commit>`, is the software reviewed
> for this paper [citation]. The fixed-seed script and machine-readable outputs
> use synthetic data and are included in that release. Software Heritage
> preserves the reviewed repository snapshot as `<reviewed-SWHID>` [citation],
> and `docs/release/<version>-release-evidence.json` records the release
> artefacts and checksums.

Every placeholder must be replaced with a verified identifier before
submission.

### P1: screening strength and scientific precision

#### 6. Lines 50–52: the uptake sentence separates effects not varied separately

Current sentence:

> In the synthetic health example, uncertainty about health gain accounted for
> more decision value than uncertainty about programme cost, and delayed or
> partial uptake increased the study size needed for positive net benefit.

The comparison changes delay and uptake together. “Delayed or partial” implies
that either factor was separately varied and shown to cause the difference.

Suggested replacement:

> In the synthetic health example, uncertainty about health gain accounted for
> more decision value than uncertainty about programme cost, and a scenario
> combining delayed and partial uptake required a larger study to produce
> positive net benefit.

#### 7. Lines 65–73: the Statement of need does not state its relation to other work

The sentences identify users and possible domains but do not explain, within
the required section, how the package relates to existing software. Add one
bounded sentence after line 69:

> Existing VOI tools provide deeper method-specific workflows; `voiage` instead
> focuses on keeping the decision description and selected results consistent
> when a workflow uses more than one language.

This sentence remains subject to the field-comparison evidence and should not
imply that all analyses already move across every binding.

#### 8. Lines 96–103: the build-versus-contribute justification remains implicit

Current paragraph explains the separate contract and its trade-off but does not
directly answer why contributing to `voi`, `BCEA`, `dampack`, or SAVI would not
meet the research requirement.

Suggested revision of lines 99–101:

> The project was developed separately because its research requirement is a
> language-neutral decision-and-result contract, whereas the compared tools
> provide method-specific R or web workflows. This contract supports workflows
> that combine language interfaces, while the established R packages retain
> greater method-specific depth.

Retain the current sentence about narrower R and Julia interfaces. It is an
important limitation.

#### 9. Lines 112–114: “numerical-parity” is unnecessary specialist language

Current sentence:

> A shared calculation reduces duplicated numerical code, while installation
> and numerical-parity tests remain necessary for each language package.

Suggested replacement:

> Sharing the calculation avoids duplicating it, but each language package
> still needs installation tests and checks that it returns the same result.

#### 10. Lines 128–130: explain why a negative EVSI estimate is diagnostic

Current sentence:

> This developing estimator uses genuine Gaussian Monte Carlo draws and returns
> the untruncated estimate so negative values remain visible as a signal to
> increase the simulation size and assess convergence.

The scientific quantity is nonnegative; a finite Monte Carlo estimate can be
negative because of simulation error. State that logic rather than treating the
sign as a generic operational signal.

Suggested replacement:

> Because EVSI is nonnegative, the developing Monte Carlo estimator retains a
> negative estimate as evidence of simulation error rather than silently
> replacing it with zero; analysts can then increase the simulation size and
> assess convergence.

#### 11. Lines 159–162: interval labels are not carried through the paragraph

The first result says “bootstrap 95% interval”, while the next sentence shortens
this to “bootstrap interval” and gives two further parenthesised intervals
without repeating the confidence level.

Suggested replacement:

> Estimated EVPI is 644 value units per person (bootstrap 95% interval 624 to
> 658); regression-based EVPPI is 590 for health gain (569 to 603) and 250 for
> programme cost (229 to 265), with 95% bootstrap intervals in parentheses.

#### 12. Lines 145–146 and 182–185: “integration” overstates the checked evidence

Current wording calls the `vop_poc_nz` artefact an integration. The immutable
commit contains a versioned `vop-voiage` contract bundle, schemas, fixtures, and
expected results. This demonstrates interoperability work, but the inspected
evidence does not establish execution of the released `voiage` package in that
workflow.

Suggested replacements:

> The repository contains one synthetic worked example and one same-author
> interoperability bundle for another research workflow.

and:

> The developer-led `vop_poc_nz` health-economic workflow contains a versioned
> interoperability bundle of schemas, fixtures, source records, and expected
> results for exchange with `voiage` [citation].

If executable cross-repository use exists, cite the exact run, test, or
published analysis instead and retain “integration”.

#### 13. Lines 186–189: honest wording reveals unresolved screening evidence

Current sentences correctly say the workflow is same-author and that no
attributable non-author use is documented. They should not be rewritten into a
stronger claim without evidence.

Before submission, obtain one or both of:

- verifiable developer research use in which the released package performed a
  real analysis, with an immutable public or editor-verifiable record; and
- attributable non-author installation, use, issue, review, or collaborative
  input.

Once evidence exists, replace lines 186–189 with the exact use, participant
relationship, resulting change, and citation. Until then, retain the candid
boundary and treat the submission as high risk at pre-review.

### P2: final-revision polish

#### 14. Line 25: submission date

Update the front-matter date only when the immutable JOSS revision is selected.
The current date is valid for this draft but cannot establish the date of a
later release or submission.

## Sentences that do not require revision

All other substantive sentences were reviewed and pass this handling-editor
round. In particular:

- the four VOI questions and expansions at lines 32–40 are accurate and
  accessible;
- the target audience and bounded cross-domain examples at lines 56–73 are
  appropriately restrained;
- the descriptions of `voi`, `BCEA`, `dampack`, SAVI, and the two Python tools
  at lines 77–94 are specific and cited;
- the analytical EVSI assumptions and the current fitted-Gaussian two-loop
  description at lines 116–134 agree with the present source and focused tests;
- the health-example assumptions and reported values at lines 145–180 agree
  with the checked machine-readable outputs;
- the funding and competing-interest statements at lines 204–207 are direct;
  and
- no promotional superlatives, vague expert attributions, generic positive
  conclusion, or conspicuous cluster of Authentext surface patterns was found.

## Checks and evidence inspected

- complete `paper.md` and `paper.bib`;
- root licence, citation metadata, README, contribution and support routes;
- Git history and live GitHub repository metadata;
- live `v1.0.0` release assets and verified tag commit;
- `docs/release/v1.0.0-release-evidence.json`;
- the Software Heritage snapshot API and its `v1.0.0` reference;
- current Rust analytical EVSI source, Python two-loop implementation, v2
  contract, and scientific tests;
- R and Julia source-binding boundaries;
- fixed-seed health-example script, figure, and machine-readable outputs;
- immutable `vop_poc_nz` contract-bundle tree at the cited commit;
- live issue #471 and its sole author-created validation request;
- hosted JOSS workflow history and workflow definition;
- current JOSS guidance linked above.

Checks run:

| Check | Result |
| --- | --- |
| `uv run python scripts/validate_joss.py` | Passed. |
| `uv run --extra ci --extra dev pytest tests/test_joss_readiness.py --no-cov -q` | 7 passed. |
| Focused EVSI and v2 contract tests | 78 passed, with expected warnings for non-stable estimators and negative Monte Carlo estimates. |
| `uv run bash scripts/vale_prose.sh paper.md` | Passed with no findings. |
| Citation-key reconciliation | 14 cited keys and 14 BibTeX records; no missing or uncited key. |
| SourceRight read-only provenance pass | Tool ran, but detected no sources because its current parser does not support this Pandoc/BibTeX input directly; no citation credit was inferred from it. |
| Hosted Open Journals build | Latest relevant run passed for committed base `1dfcd3d`; no hosted build exists for the dirty revision reviewed here. |
| Body length | 1,544 words from Summary through availability, within JOSS's 750–1,750 range. |

The repository-owned validator is useful but insufficient as an acceptance
proxy: it currently accepts the narrower AI assurance and does not require the
paper's described implementation to exist in the cited release.

## Acceptance conditions for the next handling-editor round

A later handling-editor score may exceed 995 only when all of the following are
verified at one immutable revision:

1. the paper, implementation, v2 contract, tests, Python/R/Julia manifests, and
   documentation are committed together;
2. the full required local and hosted test matrix passes that commit;
3. the reviewer-installable release contains every capability described in the
   paper;
4. release assets, checksums, provenance, SBOM, evidence manifest, and Software
   Heritage archive identify that release;
5. the Open Journals workflow builds the exact paper revision and the rendered
   PDF is visually reviewed;
6. the AI disclosure names every recoverable system/version and contains only
   human-verification assurances the author can attest;
7. demonstrated research use and the solo-project community-engagement risk are
   resolved with attributable or editor-verifiable evidence;
8. every sentence-level finding above is removed or supported; and
9. a new independent review finds no unsupported scientific, impact, release,
   archive, or policy claim.
