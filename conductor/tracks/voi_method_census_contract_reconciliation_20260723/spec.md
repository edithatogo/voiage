# VOI and VOP Method Census and Contract Reconciliation

## Overview

Maintain canonical method and Decision Problem registries that distinguish estimands, estimators, decompositions, diagnostics, applications, aliases and adjacent methods.

Owning issue: [#314](https://github.com/edithatogo/voiage/issues/314). Parent programme issue [#313](https://github.com/edithatogo/voiage/issues/313).
Project: [VOP–VOIAGE Conductor Roadmap](https://github.com/users/edithatogo/projects/28).

## Requirements

1. Freeze additive portable industry Decision Problem semantics under issue #566 without breaking accepted v1 contracts.
2. Classify residual families #593-#600 and estimation-focused variance VOI #619 before implementation, exclusion or generated capability claims.
3. Record primary definitions, equations, assumptions, aliases, evidence strength, search limits and maturity dispositions.
4. Generate hash-bound review candidates, schemas, migrations and JSON/Arrow fixtures only after classification.

## Owned issue records

[#566](https://github.com/edithatogo/voiage/issues/566)

Child issues are delivery records owned by this track. They do not become
separate Conductor tracks unless they have an independently approved contract.

## Acceptance criteria

- **AC-01:** The owning issue, native parent/children, Project 28 item,
  Conductor metadata, registry and central cross-reference manifest agree.
- **AC-02:** Estimands, contracts, aliases, maturity and unsupported states are
  explicit; planning is never advertised as installed execution.
- **AC-03:** Tests or review protocols precede implementation or positive
  parity claims and include independent references and failure cases.
- **AC-04:** Rust/Python/R/Julia/Mojo dispositions are explicit wherever the
  workstream affects executable capabilities.
- **AC-05:** Documentation, schemas, generated surfaces, examples and
  executable evidence do not exceed the reviewed maturity state.
- **AC-06:** Every current and candidate method has an evidence-backed disposition, reviewed compatibility mapping and named scientific/contract approval before registry promotion.
- **AC-07:** Automated review, full Conductor validation, repository checks and
  hosted required checks pass before repository completion.

## Non-functional constraints

- Preserve the Rust-authoritative stable core and backward-compatible v1
  contracts.
- Use deterministic, versioned, finite-validated artifacts with provenance.
- Keep optional adapters, research estimators and external systems outside the
  stable dependency boundary.
- Preserve explicit human, rights, credential, publication, registry, release
  and hosted gates.

## External and human gates

- Named scientific or contract review is required before maturity promotion.
- Hosted checks, merge, release, registry publication and external approval are
  separate from repository planning and local validation.
- Rights, privacy, practitioner or authorship confirmation remains human-owned
  where the workstream requires it.

## Out of scope

- Treating issue creation, Project status, schemas, plots, mock fixtures or
  documentation prose as runtime completion.
- Silently changing released stable numerical policy or wire contracts.
- Claiming cross-language parity without clean installed shared-fixture
  evidence.

## Authoritative inputs

- User-approved governance repair in the 2026-07-27 Codex task.
- GitHub issue [#314](https://github.com/edithatogo/voiage/issues/314) and its
  native issue hierarchy, live revision audited 2026-07-27.
- `conductor/product.md`, `conductor/product-guidelines.md`,
  `conductor/tech-stack.md`, `conductor/workflow.md`,
  `specs/v1/stable-api.json`, `roadmap.md` and `todo.md`.
- Repository default-branch baseline `cd53ce09`.
