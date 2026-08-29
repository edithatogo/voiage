# Contributor and AI Contribution Transparency

## Overview

Provide canonical human CRediT records and release-linked machine-readable AI-assistance provenance with accountable human review.

Owning issue: [#323](https://github.com/edithatogo/voiage/issues/323). Parent programme issue [#313](https://github.com/edithatogo/voiage/issues/313).
Project: [VOP–VOIAGE Conductor Roadmap](https://github.com/users/edithatogo/projects/28).

## Requirements

1. Create synchronized contributor, CRediT, citation and codemeta records without inferring authorship from commit counts.
2. Create a versioned AI contribution schema, append-only ledger and manuscript disclosure with provider/tool/model when known, purpose, disposition, verification and limitations.
3. Exclude chain-of-thought, secrets, confidential data and raw prompts by default; AI is not an author or CRediT contributor.
4. Validate referential integrity, release scope, review state, model identity, claim hygiene and manuscript synchronization.

## Owned issue records

No native child issues; tasks remain within the owning issue.

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
- **AC-06:** Generated text is insufficient; human role assignments and material AI-assisted contributions require accountable human confirmation.
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
- GitHub issue [#323](https://github.com/edithatogo/voiage/issues/323) and its
  native issue hierarchy, live revision audited 2026-07-27.
- `conductor/product.md`, `conductor/product-guidelines.md`,
  `conductor/tech-stack.md`, `conductor/workflow.md`,
  `specs/v1/stable-api.json`, `roadmap.md` and `todo.md`.
- Repository default-branch baseline `cd53ce09`.
