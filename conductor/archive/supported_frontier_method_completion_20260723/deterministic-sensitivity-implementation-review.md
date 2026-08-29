# Deterministic sensitivity implementation review

Date: 2026-08-01

Scope: issue #556 and delivery subissues #724--#728 on draft PR #723

Review range: `5859f7a1..cb20640f`, followed by the F556-5 remediation diff

## Initial independent result

The first independent implementation/contract review failed with two High and
five Medium findings. It found a false full-Cartesian declaration path, a CLI
registry integration concern, duplicate/unknown coordinate paths, unordered
switch grids, an omitted serialized rank, incomplete promised pathology and
property tests, and incomplete digest evidence. No Critical or security
finding was identified.

## Remediation

- `full-cartesian-independent` now requires the exact Cartesian point set.
- Normalized records, scenario coordinates, two-way surfaces and feasible
  points have unique identities; unused records and unknown surface keys fail
  closed.
- One-way grids must be strictly increasing, so switch brackets preserve their
  lower/upper meaning.
- Competition rank is present in the runtime result, wire schema and normative
  fixture.
- Brute-force/property, multiple/no-switch, discontinuity, plateau,
  tolerance-boundary, repeatability and immutability cases are executable.
- The exact evidence set pins the installed contract, CLI, runtime, plot, all
  DSA test layers, schemas, fixture and documentation.
- M18 is synchronized into the local canonical C16 projection, and the Python
  module, extension surface and runtime inventory are explicit without eager
  optional/heavy imports.

## Re-review result

PASS. The independent reviewer verified every original finding against the
corrected worktree and reported no remaining Critical, High, Medium or Low
implementation-review findings. Focused DSA, CLI registry, export, Ruff, ty,
frontier, cross-reference, C16 and adversarial checks passed. No platform guide
was selected by the track manifest; platform-guide review is therefore not
applicable to these Python, JSON, Markdown and Mermaid paths.

The complete Python suite also passed after remediation: 2,962 passed, 15
skipped, with 92.61% aggregate coverage. The 100 reported warnings are the
repository's existing warning surface and did not include a test failure.

## Boundary

This is an independent implementation and contract review, not named
scientific approval. Hosted exact-head checks, merge, Rust/R/Julia execution,
stable promotion, release and issue closure remain separate gates.
