# Track Specification: Starlight Documentation Platform

## Overview
This track defines the documentation-platform strategy for `voiage` if the
project adopts Starlight as a versioned docs site. The goal is to make the
Starlight stack explicit before any migration work begins, including which
Starlight version family to pin, which plugins are required, which plugins are
optional, and how the current Sphinx-based documentation remains the source of
truth during transition.

## Functional Requirements
1. The track must record the intended Starlight adoption boundary:
   - docs-site platform for future migration work
   - versioned docs surface with explicit release/update policy
   - no silent replacement of the current Sphinx docs during planning
2. The track must define the Starlight versioning policy, including how the
   version pin will be updated and documented.
3. The track must define the plugin baseline for a docs site built on
   Starlight:
   - required: `starlight-versions`, `starlight-links-validator`
   - conditional/likely: `starlight-image-zoom`, `starlight-heading-badges`,
     `starlight-sidebar-topics`, `starlight-utils`
   - search integration remains explicit; keep Pagefind unless an external
     search provider is justified later
4. The track must define the docs-content structure that versioned Starlight
   pages would use, including how edit links, version groups, and release
   branches are expected to work.
5. The track must update repo documentation and conductor setup so the roadmap,
   tech-stack, product notes, todo list, and changelog all reflect the chosen
   Starlight strategy.
6. The track must define the validation gates that future implementation work
   would need to satisfy:
   - site build
   - docs lint / prose lint
   - link validation
   - version-aware navigation checks
   - content smoke tests for migrated pages

## Non-Functional Requirements
1. The track must keep the current Sphinx docs and existing docs workflow
   intact until a later implementation track explicitly changes the primary
   documentation site.
2. The plugin list must stay conservative and justified by concrete docs needs.
3. The resulting roadmap and setup notes must be explicit enough that a later
   implementation track can execute without reopening the Starlight baseline
   decision.

## Acceptance Criteria
1. A dedicated conductor track exists for the Starlight docs platform.
2. The roadmap includes a Starlight phase or equivalent roadmap item.
3. The conductor setup documents record the versioning and plugin baseline.
4. The track documents which plugins are required, optional, and conditional.
5. The future validation and migration handoff requirements are written down.

## Out of Scope
1. Building the Starlight site itself.
2. Migrating the current docs tree into Starlight.
3. Changing the current Sphinx documentation pipeline unless a later track
   explicitly does so.
