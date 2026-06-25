SOTA Strategy Orchestration and Dependency Matrix
=================================================

This guide turns the Phase 11 strategy work into a dependency-aware execution
model that can be split cleanly across subagents. The goal is to keep the
packaging, HPC, Rust-core, and docs work parallelizable without losing the
shared decisions that have to happen first.

Why this exists
---------------

The repository already has four strategic workstreams:

* packaging review readiness
* HPC distribution and acceleration strategy
* Rust-core ABI and migration strategy
* polyglot repo and documentation architecture

Those workstreams share a few decisions and artifacts. If those shared items
are not spelled out, later implementation work will either duplicate effort or
make incompatible assumptions.

Shared decisions that should be treated as prerequisites
--------------------------------------------------------

1. Release and publishing playbooks per language binding

   * define the registry target, provenance, rollback, and version-sync rules
   * keep this separate from the individual binding implementation tracks

2. Rust-core compatibility matrix

   * define which binding versions are compatible with which Rust core versions
   * record the no-breaking-change rule for the current Python/R/Julia/TS/Go/.NET APIs

3. Benchmark and accelerator evidence gates

   * require measured evidence before any GPU, TPU, or custom-circuit claim
   * keep HPC distribution separate from accelerator promotion

4. Documentation navigation and versioning rules

   * decide when a docs-version switch is justified
   * keep the current Sphinx docs authoritative until a later migration track changes that

5. Distribution metadata and review checklists

   * keep pyOpenSci, rOpenSci, JOSS, Spack, EasyBuild, and community-alignment artifacts explicit

6. Registry deployment before HPC-native escalation

   * finish the language release submission program before treating HPC-native
     work as the next implementation stage

Parallel lane map
-----------------

The lanes below can proceed in parallel once the shared prerequisites are in
place.

.. list-table::
   :header-rows: 1

   * - Lane
     - Blocking prerequisite
     - Safe to parallelize with
     - Required output
   * - Packaging review readiness
     - release playbook + metadata
     - HPC, docs, ABI
     - fit matrix + repo change checklist
   * - HPC distribution and acceleration
     - release artifact policy
     - packaging, docs
     - distro matrix + accelerator ranking
   * - Rust-core ABI and migration
     - stable contracts + versioning
     - packaging, docs, HPC
     - ABI recommendation + compat matrix
   * - Polyglot repo and documentation
     - docs inventory + navigation
     - packaging, HPC, ABI
     - future repo/docs layout + migration

Recommended execution order
---------------------------

1. Lock the shared prerequisites.
2. Run the four lanes in parallel.
3. Review the lane outputs for conflicts.
4. Turn the reviewed outputs into implementation tracks or repo changes.

Release and publishing playbook
-------------------------------

Every language binding should have the same minimum release story:

* canonical version source
* registry target
* provenance or signature story
* rollback path
* release notes or changelog update
* package dry-run validation in CI

Compatibility matrix
--------------------

The migration path should keep the existing public APIs stable while the core
moves to Rust.

* Python remains the primary façade.
* R, Julia, TypeScript, Go, and .NET remain thin adapters.
* A narrow C ABI is optional and only worth adopting if the native bridge
  value is clear.
* TypeScript should prefer WASM or N-API over a broad C ABI if the JS edge
  needs a native backend.

HPC and accelerator gates
-------------------------

* Classify the library as HPC-deployable, HPC-friendly, or HPC-native before
  any registry claim is made.
* Require install/recipe evidence for Spack and EasyBuild before claiming
  distribution readiness.
* Treat HPSF and E4S as ecosystem/curation targets, not a substitute for
  package recipes.
* Promote accelerator claims only after benchmark and workload evidence has
  been recorded.

Documentation architecture rules
---------------------------------

* Keep the current Sphinx docs authoritative until a later migration track
  explicitly changes the primary site.
* Organize the future docs around current users, language targets, and release
  channels rather than around the implementation language alone.
* Keep tutorial and binding walkthrough surfaces aligned to the same canonical
  use cases.

Subagent partitioning
---------------------

A future implementation can split work cleanly as follows:

* one agent for packaging/community review readiness
* one agent for HPC distribution and accelerator evidence
* one agent for Rust-core ABI and compatibility policy
* one agent for docs architecture and versioning
* one synthesis agent for dependency conflicts and final handoff

Dependency sketch
-----------------

.. code-block:: text

   Current state -> Shared prerequisites -> Parallel lanes -> Reviewed outputs -> Implementation tracks

.. code-block:: text

   Python facade -> Rust core -> optional C ABI edge
   Rust core -> stable contracts -> language adapters
   Contracts -> fixtures -> CI/release gates
