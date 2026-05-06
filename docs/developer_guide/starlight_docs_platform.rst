Starlight Documentation Platform
================================

This note records the current docs-platform strategy for `voiage` in case the
project later migrates from Sphinx to a Starlight-based site.

Baseline Decision
-----------------

* Starlight is the candidate docs-site platform, not a silent replacement for
  the current Sphinx docs.
* The platform plan should pin a specific Starlight version family and record
  upgrade/update steps explicitly when the version changes.
* The default search path should stay on Pagefind unless a later decision
  justifies a different provider.

Required Plugins
----------------

* `starlight-versions` for versioned docs pages and release-aligned navigation.
* `starlight-links-validator` for broken-link validation in the docs site.

Conditional Plugins
-------------------

These are useful if the eventual Starlight site needs the corresponding UX:

* `starlight-image-zoom`
* `starlight-heading-badges`
* `starlight-sidebar-topics`
* `starlight-utils`

Migration Boundary
------------------

The current Sphinx documentation remains authoritative until a later
implementation track explicitly changes the primary docs site. Any future
Starlight migration must preserve:

* the current docs content hierarchy
* version-aware navigation
* link validation
* docs build and content smoke checks

This note is intentionally conservative so later implementation work can follow
the recorded baseline without reopening the platform decision.
