# Scientific review panel and orchestration protocol

Experimental-frontier scientific review uses a panel of role-specific
subagents coordinated by a separate orchestrating agent. The panel produces
challenge evidence and a promotion-readiness recommendation. It does not
authorize stable promotion, publication, registry submission, release or issue
closure; those decisions remain separate accountable gates.

## Roles and independence

The **orchestrating agent** freezes the candidate, commissions reviewers,
normalizes reports, challenges unsupported claims, maintains the finding and
disagreement registers, and synthesizes recommendations. It does not review its
own synthesis, vote away dissent, remediate findings, or approve promotion.

Each family requires reports from at least four specialist roles:

1. **Estimand and domain reviewer** — definitions, conditioning, assumptions,
   units, utility, costs, population, horizon, ethics and limiting cases.
2. **Estimator-assurance reviewer** — reference calculations, bias, RMSE,
   coverage, calibration, convergence, uncertainty, pathologies and
   reproducibility.
3. **Cross-language and API reviewer** — schemas, fixtures, installed-wheel
   execution, Rust/Python/R/Julia/Mojo dispositions, compatibility and
   capability discovery.
4. **Governance and publication reviewer** — reviewer eligibility, conflicts,
   evidence integrity, claims, issue/Project/roadmap traceability,
   adjudication and publication boundaries.

Add another qualified domain reviewer when clinical, environmental, financial,
equity, safety or regulatory claims require specialist expertise. Every
reviewer records identity, role, relevant qualifications, prior contribution,
conflicts and an independence attestation. An author or remediator may explain
the work but cannot independently approve that slice. If a reviewer remediates
a finding, a fresh eligible reviewer re-reviews it.

Automated or subagent reports are structured challenge evidence. In this
single-person repository, scientific and domain advice is delegated to
role-separated agents and the repository owner records the accountable
scientific decision. The owner decision is not independent review and must not
be described as such. External venues retain any reviewer requirements they
control.

## Review packet and entry criteria

The orchestrator must reject a dirty, moving or incomplete candidate. Freeze an
immutable packet containing:

- exact commit and tree identifiers plus an artifact SHA-256 manifest;
- dependency lock, toolchain, platform and deterministic command record;
- estimands, schemas, fixtures, algorithms, tests, documentation and claims;
- issue hierarchy, Project fields, roadmap, todo and canonical projections;
- reference identifiers, independently reconstructed calculations and known
  limitations;
- language/runtime dispositions and prior review/finding history.

The packet is valid only for the frozen candidate. A later scientific,
numerical, runtime, schema, fixture or claim change invalidates the affected
approval. Every delta invalidates approval by default. A deterministic allowlist
may classify a metadata-only delta for bounded review only when both the
governance reviewer and an affected scientific reviewer independently sign the
classification, hashes and rationale.

## Review waves

Run reviews in risk and dependency order. Reviewers submit their reports before
seeing the orchestrator's consolidated adjudication.

1. **Wave A — specialized v1.2.0 families:** #619 estimation variance, #571
   COSS/ENBS/efficiency and #595 expected utility/VoC.
2. **Wave B — high-risk v1.3.0 families:** #570 risk-sensitive/constrained,
   #599 heterogeneity and sparse subgroups, #600 outcome-conditional sample
   information, #597 sequential belief-state and #598 signed/social value.
3. **Wave C — remaining C17/C18 families:** #556–#560, #572, #582 and
   #593–#596, grouped only where estimands and evidence remain separable.
4. **Cross-cutting wave:** installed artifacts, portable fixtures, capability
   discovery, language parity, reproducibility and stable-promotion evidence.

Finding remediation occurs in issue-backed implementation slices, not inside
the scientific-review verdict. Each new candidate is rebound to a fresh packet
and receives the affected role reviews again.

## Reviewer rubric and acceptance matrix

For every applicable row, report `Pass`, `Minor revision`, `Major revision`,
`Fail`, or `Not applicable` with a rationale. No total score or majority vote
may mask a failed required row.

| Dimension | Minimum acceptance evidence |
| --- | --- |
| Estimand | Target, decision/estimation focus, conditioning, comparator and interpretation are mathematically explicit. |
| Units and scope | Units, population, horizon, discounting, utility/cost placement, stakeholder and comparability are coherent. |
| Assumptions | Identification, independence, regularity, feasibility and perfect/imperfect-information assumptions are testable. |
| References | Equations and algorithms map to primary or authoritative sources and an independent reconstruction. |
| Numerical validity | Bias, RMSE, coverage/calibration, convergence, tolerances, ties, bounds, degeneracy and pathologies are addressed. |
| Sensitivity | Conclusions are tested across scientifically relevant priors, designs, utilities, models and grids. |
| Uncertainty | Nested-estimator uncertainty, Monte Carlo error and uncertainty around selected optima are reported without overclaiming. |
| Vector targets | Covariance functional, units and any scalarization are declared and unit-safe. |
| COSS commissioning | No-study comparison is explicit, or the result is unambiguously labelled as conditional on commissioning. |
| Sampling-process harm | Any harm caused by sampling has an explicit estimand and evidence, or is declared unsupported and separately scoped. |
| Reproducibility | An independent reviewer can install, replay seeds, validate fixtures and reproduce results from the packet. |
| Contract integrity | Schemas, semantic validation, serialization, provenance, versions and reconstructed result relations agree. |
| Maturity and claims | API, docs, registry, issue, Project, roadmap and publication language stay within the evidence. |
| Parity | Installed language implementations consume the same normative fixtures and tolerances, or return governed unsupported states. |
| Ethics and domain | Equity, safety, consent, regulatory, environmental and financial risks are addressed where applicable. |
| Evidence integrity | Revision, hashes, commands, identities, conflicts, reports, findings, dissent and adjudication are complete and immutable. |

Automatic failure conditions are an unidentified or conflicted approving
reviewer, missing revision or artifact digest, failed independent reproduction,
unsupported stable/parity/publication claim, stale packet, missing required
domain reviewer, unresolved Critical/High finding, disputed scientific validity,
or approval represented only as a Boolean.

## Finding, disagreement and adjudication rules

Every finding records an identifier, severity, affected estimand or claim,
exact artifact, evidence, reproducible case, required disposition, owner,
dependency and promotion impact. Findings are never silently deleted or
downgraded. Allowed dispositions are `fixed`, `reviewed_exclusion`,
`accepted_experimental_risk`, `deferred_with_explicit_exclusion`, or `disputed`;
the latter four require a rationale and accountable owner.

- A Critical or High finding may be dispositioned only as independently
  verified `fixed` or `reviewed_exclusion`. The excluded capability cannot be
  promoted. Any other Critical/High disposition blocks scientific acceptance
  and stable promotion.
- Every Medium finding requires reviewer and maintainer disposition plus
  affected-role re-review.
- Low findings may enter a versioned backlog only when they cannot affect
  validity, reproducibility or claims.
- Scientific-validity dissent keeps the family experimental. The orchestrator
  records both positions and the evidence needed to resolve them; it cannot
  settle the dispute by majority vote.

The orchestrator produces a synthesis that distinguishes: already governed
pending gates, newly discovered repository defects, separately scoped research
questions, external human decisions and release/publication gates.

## Required outputs and verdicts

The append-only evidence set comprises `review-packet.json`,
`artifact-manifest.json`, `reviewer-attestations.json`, one signed report per
role, `finding-dispositions.json`, `disagreement-register.json`,
`orchestrator-synthesis.md`, `adjudication.json`, `scientific-approval.json`
and, only when separately authorized, `promotion-receipt.json` and
release/publication receipts.

Every adjudication or approval receipt requires the adjudicator/approver's
identity, role, qualifications, conflict and independence status; exact
candidate commit/tree and packet hash; family and capability scope; decision,
conditions and dissent references; issue and evidence references; decision
date, expiry and supersession link. Adjudication is synthesized by an agent
orchestrator separated from reviewing and remediating roles. The repository
owner makes the accountable scientific and maturity decisions after reviewing
the synthesis and dissent.

The panel may recommend only:

- `scientifically_acceptable_experimental`;
- `conditional_remediation_and_rereview`;
- `major_revision_and_full_rereview`; or
- `reviewed_exclusion`.

The maintainer separately records the product/maturity decision. Scientific
acceptance does not imply installed parity, promotion, hosted exact-head
assurance, release, publication, registry acceptance or issue closure.

## Human-confirmation execution contract

Before soliciting a human decision, the candidate must be `candidate_frozen`,
not a moving preparation record. Git commit and tree OIDs use their declared
Git object format; packet, manifest and artifact digests use SHA-256 over
documented canonical bytes. Validators recompute every digest, verify manifest
members against the frozen tree and reconcile the complete prior finding and
disagreement inventory. Preparation templates use a distinct filename and
schema and cannot enter a normative review bundle.

Role reports must match the reviewer's attestation for identity, scope,
qualifications, conflicts, contribution and remediation history. The
orchestrator, author/remediator, reviewing agents, chair and accountable owner
are separated according to the family risk. Issue #850 additionally requires
distinct scientific and domain/ethics agent roles. The repository owner remains
the single accountable decision-maker. A separate agent chair is required for
disputed findings, scientific dissent, or reviewer remediation.

Accepted human receipt channels are a verified signed commit, an authenticated
GitHub review or comment with immutable URL/event identifier and body digest,
or an external authoritative receipt URI and digest. Each receipt records the
verification method and signed payload digest without retaining credentials or
secrets. Expired, withdrawn, superseded or materially stale receipts cannot be
relied upon.

The orchestrator's synthesis preserves all reports and dissent and presents
options, contingencies, rationale and an evidence-backed recommendation. It is
not an approval. Any Critical/High finding requires an independently verified
fix or reviewed exclusion; every Medium requires disposition and affected-role
re-review. A reviewed exclusion binds the omitted capability scope. Delta
review is limited to narrowly enumerated administrative fields; changes to
specifications, contracts, schemas, fixtures, estimators, claims, review
evidence or executable documentation trigger affected full review.

State changes occur in order: preparation, frozen candidate, review,
remediation and re-freeze when needed, scientific outcome, separate maintainer
decision, downstream parity/promotion/release gates, then issue closure. After
any GitHub or Project mutation, read back the exact values before appending
Conductor evidence. A partial mutation sets Sync State to `Conflict`, keeps the
gate pending and prohibits closure.
