# Track Specification: Comprehensive VOI Software Landscape And Improvement Review

## Overview

Create the next review-first programme: identify discoverable open-source,
commercial, hosted, archived, and adjacent Value of Information software; map
each product at schema, feature, subfeature, option, workflow, and adoption
levels; compare it with VOIAGE; and present evidence-linked improvements for
human review before any later roadmap incorporation.

This refines the existing external-library parity track rather than creating a
duplicate census. The current 27-tool snapshot is a reusable baseline, not a
claim that the expanded review is complete.

## Authoritative inputs

- `specs/software-landscape/` registries, schemas, evidence, generated matrix,
  and routed gap report at the reviewed branch revision.
- `conductor/requirements.md`, `conductor/design.md`, and the canonical method
  registry and DecisionProblem contracts.
- GitHub parent [#315](https://github.com/edithatogo/voiage/issues/315) and
  native subissues
  [#569](https://github.com/edithatogo/voiage/issues/569),
  [#565](https://github.com/edithatogo/voiage/issues/565),
  [#568](https://github.com/edithatogo/voiage/issues/568),
  [#573](https://github.com/edithatogo/voiage/issues/573), and
  [#567](https://github.com/edithatogo/voiage/issues/567).
- Version-pinned package registries, repositories, documentation, software
  papers, supplements, archived releases, and publicly observable commercial
  product material collected by the track.

## MoSCoW requirements

### Must have

1. [#569](https://github.com/edithatogo/voiage/issues/569),
   `landscape-schema-review-protocol`, freezes the inclusion, exclusion,
   discovery, evidence, rights, freshness, duplicate, and review protocol
   before the expanded inventory is treated as comprehensive.
2. For every product and reviewed version, record:
   - identity, category, maintenance, license, availability, pricing or
     observability, platform, languages, install/deployment, and provenance;
   - data, model, uncertainty, decision, information-action, utility,
     risk/constraint, study/design, result, and report schemas;
   - estimands, estimators, algorithms, APIs, commands, features, subfeatures,
     options, defaults, diagnostics, errors, plots, reports, examples, tests,
     workflows, integrations, interoperability, performance and accessibility;
   - authoritative evidence, extraction coverage and limitations, review date,
     review due, and evidence strength.
3. [#565](https://github.com/edithatogo/voiage/issues/565),
   `landscape-open-source-inventory`, searches language registries, source
   hosts, archives, papers, supplements, HTA and decision-analysis software,
   Bayesian OED, active learning, causal policy learning, forecasting,
   optimization, and information economics. Versions or commits are pinned.
4. [#568](https://github.com/edithatogo/voiage/issues/568),
   `landscape-commercial-hosted-inventory`, records only publicly observable
   commercial, proprietary, spreadsheet, web, and hosted behavior. It never
   infers hidden algorithms, defaults, numerical equivalence, security, or
   performance.
5. [#573](https://github.com/edithatogo/voiage/issues/573),
   `landscape-capability-adoption-map`, maps every observed capability to
   canonical identifiers and `native`, `equivalent`, `adapter`, `planned`,
   `excluded`, or `not-reproducible`. It also captures useful onboarding,
   workflow, interaction, export, reporting, collaboration, governance,
   deployment, and enterprise-integration lessons.
6. [#567](https://github.com/edithatogo/voiage/issues/567),
   `landscape-gap-review-roadmap-proposal`, produces a deterministic proposal
   covering missing methods, schemas, options, diagnostics, UX, reporting,
   examples, integrations, performance, assurance, governance, and reviewed
   exclusions. Each recommendation records user value, target roles/domains,
   novelty, evidence, dependencies, design, license risk, MoSCoW, priority,
   effort, maturity, owning track, proposed subissue, alternatives, and
   approved/rejected/deferred state.
7. The source census, generated comparison, gap analysis, and human decision
   ledger remain separate artifacts. Discovery cannot silently promote a
   method, overwrite scientific notes, or change the roadmap.
8. Every positive parity claim names independent, competitor-free fixtures and
   executable tests. VOIAGE remains usable with competitor packages absent.
9. The inventory explicitly searches for the residual #593--#600 families and
   submethods: EVPIM, EVSIM (specific implementation), EVP, EVEIm/EVSEIm
   terminology and implementation-adjusted EVSI; EVIU,
   EEV/VSS/wait-and-see/DVSS/VMS; EUI/CEI/buying/selling prices; event and
   tail-event information plus information density; POMDP observation value,
   adaptive management and dual control; negative/social/team information
   value; static/dynamic heterogeneity; and delta-EV/VSI/sigma-VSI/rVSI.
   Observations map to the planning register until additive scientific review.

### Should have

- Inspect source, tests, examples, vignettes, schemas, changelogs, issue
  histories, release artifacts, manuals, tutorials, demonstrations, and
  integration guidance where legitimately available.
- Preserve archived or inactive software when it offers a unique method or
  adoption lesson, with maintenance state and closest supported workflow.
- Generate views by product, method, capability class, ecosystem, domain,
  parity state, evidence strength, maintenance, license, adoption lesson,
  MoSCoW, priority, risk, and review due.
- Produce an accessible maintainer review packet with concise recommendations
  plus drill-down evidence.

### Could have

- Candidate submissions using schema-validated metadata and duplicate
  detection.
- Automated metadata refresh proposals from authoritative endpoints.
- Comparative migration examples for source-shaped schemas with demonstrated
  user value and independently constructed fixtures.

### Won't have now

- A universally exhaustive claim.
- API or trademark imitation, incompatible source copying, or competitor
  runtime dependencies in stable workflows.
- Hidden implementation, performance, security, or numerical-parity claims for
  proprietary, web-only, inaccessible, or documentation-only products.
- Automatic acceptance of recommendations, issue closure, maturity promotion,
  or roadmap changes.

## Data model and evidence policy

The landscape schema must support products, versions, artifacts, nested
capabilities, schemas, options/defaults, evidence observations, license/rights
records, parity dispositions, adoption lessons, gaps, proposals, decisions, and
refresh history. Each derived record retains stable source identifiers so that
the comparison can be regenerated without erasing reviewer annotations.

Evidence strength is ordered from executable version-pinned source/tests,
through version-pinned documentation and observable demonstrations, to
inaccessible or not reproducible. A lower evidence state cannot support a
higher parity claim.

## Acceptance criteria

1. #569, #565, #568, #573, and #567 satisfy their issue contracts and are
   native subissues of #315 with complete Project 28 metadata.
2. Every included tool/version and capability is schema-valid, evidence-linked,
   review-dated, and explicitly bounded.
3. The open-source and commercial inventories, feature/adoption map, gap
   report, and improvement proposal regenerate deterministically.
4. Every proposed improvement routes to an existing track or a reviewed new
   owner without creating duplicate issues.
5. Every observed #593--#600 capability maps to its candidate record or a
   reviewed alias/application/adjacent disposition without promoting the
   candidate register to canonical status.
6. The maintainer can approve, reject, revise, or defer proposals individually;
   the source inventory remains unchanged.
7. No roadmap change is represented as approved until the named review is
   recorded in a later, checksum-bound change.

## Non-functional constraints

The review must be reproducible, license-aware, privacy-preserving,
competitor-independent, bounded by recorded search dates, and refreshable
within 93 days and before a minor release. Network discovery is never required
for deterministic tests.

## External and human gates

- Public observability limits commercial and hosted claims.
- Source availability, archival access, licensing, and paid-product access are
  external and may remain blocked.
- The improvement proposal requires named maintainer review before roadmap
  incorporation.

## Out of scope

Implementing the proposed improvements during the landscape review, purchasing
software, accepting licenses, accessing private products, or publishing a
competitive superiority claim.
