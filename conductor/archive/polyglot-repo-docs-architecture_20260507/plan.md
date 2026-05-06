# Track Implementation Plan: Polyglot Repo And Documentation Architecture

## Phase 1: Inventory The Current Structure [checkpoint: complete]

- [x] Inventory finding: the repo is already cleanly split into `voiage/`,
  `bindings/`, `r-package/`, `docs/`, `examples/`, `specs/`, `conductor/`,
  `tests/`, `scripts/`, and `validation/`, with the binding trees already
  acting as language-specific release surfaces.
- [x] Inventory finding: the docs are organized by audience and workflow in
  practice, with Sphinx user/developer/method/reference sections, notebook and
  walkthrough surfaces, release notes, and separate ecosystem/frontier docs.
- [x] Inventory finding: the current patterns that already work are the
  language-specific README/tutorial entry points, the Sphinx docs index, the
  release matrix, and the Conductor track registry.

## Phase 2: Design The Future Repo And Docs Layout [checkpoint: complete]

- [x] Design finding: the best future layout is a named core plus thin
  language adapters, with `voiage/` as the Python façade, `bindings/rust/` as
  the canonical engine, and the other binding directories remaining
  language-owned adapter packages.
- [x] Design finding: docs should stay organized by user workflow first, then
  core concepts, then language-specific walkthroughs, with release/governance
  material separated into explicit release and developer-guide sections.
- [x] Design finding: shared canonical use cases should be authored once in the
  docs/tutorial index, then referenced from each language walkthrough rather
  than duplicated per binding.

## Phase 3: Define The Migration And Navigation Plan [checkpoint: complete]

- [x] Migration finding: the safest order is to keep current landing pages
  stable, promote the core/release/governance split in docs, and only then
  move or rename pages if a future track proves the gain is worth the link
  churn.
- [x] Navigation finding: front-door pages should continue to guide users by
  workflow and language, with index pages preserving current entry points
  during any transition.
- [x] Governance finding: future tracks should keep governance and release
  rules in dedicated docs sections and avoid scattering them across tutorial or
  API reference pages.
