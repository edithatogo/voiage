# JOSS panel synthesis: round 2

## Decision

Round 2 requires major revision. The panel score is **824/1000**, the lowest
complete reviewer score. This is not an acceptance prediction or an editorial
decision.

| Role | Score |
| --- | ---: |
| Editor-in-chief screening | 852 |
| Handling editor | 918 |
| Domain reviewer | 886 |
| Research-software reviewer | 866 |
| Reproducibility reviewer | 891 |
| Numerical verifier | 890 |
| Applied-research accessibility reviewer | 824 |
| Authentext sentence editor | 884 |

All eight reviewers inspected the full paper. The accessibility and sentence
roles classified every substantive sentence. The reports agree that the
manuscript's restrained tone, four-question opening, candid adoption boundary,
and fixed-seed health example are substantial improvements over round 1.

## Convergent findings

### 1. Scientific study-model contract

The numerical reviewer found that the then-current generic two-loop EVSI path
discarded correlation in the joint prior. This produced a material numerical
error in an independent conjugate test. The panel therefore required either a
narrow enforced prior contract or explicit study-simulation and joint-posterior
callbacks. The paper also needed to distinguish the analytical study model from
non-stable compatibility estimators.

### 2. Worked-example interpretation

The numerical, domain, accessibility, and sentence reviewers independently
identified the same omission: the paper reported ENBS thresholds without
stating the population, horizon, discounting, study-cost, delay, and uptake
assumptions. They also required:

- the cost distribution and independence assumption;
- an explicit statement that the study informs health gain, not programme
  cost;
- identification of EVPI and EVPPI as finite-simulation estimates;
- the regression basis of EVPPI;
- uncertainty intervals for simulation estimates;
- the sample size associated with analytical EVSI; and
- bounded language that treats the synthetic example as an illustration, not
  a real-trial estimate.

### 3. Release and archival identity

Every editorial or reproducibility role found that public release `v1.0.0`
predates the revised EVSI and binding work. A final paper cannot cite that
release as the software containing the new behaviour, promise a future freeze,
or imply that its Software Heritage identifier covers later changes. The
reviewed software requires an immutable release, hosted evidence, a matching
archive, and a factual availability statement.

### 4. Reviewer-installable packaging

The research-software and reproducibility reviewers found four related
repository defects:

- the ordinary source-distribution path did not preserve immutable source
  identity;
- top-level Python import eagerly loaded optional scientific stacks;
- source-local R documentation tests depended on the caller's location; and
- Rust Clippy and workspace verification were not green on the reviewed
  snapshot.

The reviewer instructions also needed exact R and Julia build commands and a
fully specified second independent-validation exercise.

### 5. Evidence scope

The existing CycloneDX file described the Python environment rather than the
complete mixed-language workspace. The paper must either label that scope
precisely or cite a verified composed SBOM. Developer-led use in
`vop_poc_nz` meets the minimum research-use interpretation only when described
as same-author use and pinned to immutable evidence. It is not independent
adoption.

### 6. Citations and field positioning

The panel required:

- correction of James F. Murray and Gillian D. Sanders Schmidler in the
  Rothery record;
- the Tuffaha et al. review of web-based VOI tools;
- a Python comparator;
- immutable citations for software and cross-project evidence; and
- a simpler build-versus-contribute rationale that explains the research
  consequence rather than internal interface terminology.

### 7. Sentence-level accessibility

The sentence roles found low promotional or formulaic-language density, but
identified repeated abstract subjects, long noun catalogues, unexplained
engineering terms, and inconsistent estimator status. They prioritised plain
descriptions of what researchers can transfer, which assumptions determine a
study result, and how population, timing, uptake, and cost change research
value.

## Prioritised remediation

1. Replace inferred generic two-loop updating with explicit simulation and
   joint-posterior callbacks, and add correlated-prior regression tests.
2. Make the worked example self-contained and report rounded estimates,
   uncertainty intervals, analytical values, evaluated ENBS signs, and complete
   assumptions.
3. Correct and freeze all citation records; add the web-tool review and
   directly relevant Python alternatives.
4. Align the manuscript, API documentation, examples, and warnings with the
   scientific maturity boundary.
5. Repair clean source builds, bounded imports, Rust checks, R/Julia reviewer
   paths, and complete SBOM evidence.
6. Publish and archive the exact reviewed software revision, then replace all
   prospective availability language with observed identifiers.
7. Build the official JOSS PDF from that immutable revision and repeat the
   eight-role review with a fresh sentence inventory.

## Gate separation

Manuscript defects, repository defects, and external outcomes remain separate.
Attributable non-author use, arXiv announcement, JOSS submission, editorial
screening, review, acceptance, and DOI assignment cannot be manufactured by
the internal panel. The paper can pass internal manuscript review while
accurately recording an external gate as pending, but it cannot claim that the
gate has been satisfied.

## Round 3 entry conditions

Round 3 begins only after the remediation above is implemented and verified.
It must inspect one committed revision, the official rendered PDF, release and
archive evidence, reviewer installation paths, citations, figures, and every
substantive sentence. The requested threshold is met only if every complete
review scores at least 996 and records no manuscript blocker.
