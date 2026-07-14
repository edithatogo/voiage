# Track Specification: Polyglot Repo And Documentation Architecture

## Overview
This track decides whether the repository and documentation should be
reorganized for a Rust-core, polyglot-scientific-library future. The question
is not whether the repo is broken. The question is whether clearer separation
of core, bindings, docs, tutorials, releases, and community guidance would make
the project easier to maintain and easier to submit to the target communities.

## Functional Requirements
1. The track must inventory the current repository and documentation layout.
2. The track must propose a future-state organization for:
   - core implementation
   - bindings and adapters
   - docs and tutorials
   - specs and fixtures
   - release and distribution docs
   - community and governance docs
3. The track must determine whether the docs should be organized around:
   - user workflows
   - language targets
   - core concepts
   - release channels
   - a mix of the above
4. The track must define how the current Sphinx docs, binding READMEs, and
   roadmap/Conductor material should relate to one another.
5. The track must define a migration order that can be executed without
   breaking links or leaving users without a clear landing page.

## Non-Functional Requirements
1. The proposed structure must improve clarity without forcing a large-scale
   move unless the gain is real.
2. The documentation structure must support a polyglot scientific library,
   not just a Python-centric package with extra directories.
3. The track must keep the current docs authoritative until a later migration
   track explicitly changes the primary site.

## Acceptance Criteria
1. A future-state repo layout is described.
2. A future-state docs layout is described.
3. The relationship between core docs, binding docs, tutorials, and release
   docs is explicit.
4. A migration order is written down so later work can move safely.

## Out of Scope
1. Moving the repo to a new layout immediately.
2. Migrating the docs site generator in this track.
3. Breaking existing links or current documentation access patterns.
