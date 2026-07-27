# Study-Design Efficiency, COSS and Experiment-Portfolio VOI

## Overview

Extend the existing experiment-portfolio plan with governed single-study
efficiency outputs: a Curve of Optimal Sample Size (COSS) based on signed ENBS
and a dimensionless EVSI/EVPI diagnostic. Reconcile the existing plotting and
legacy clinical-optimizer helpers without promoting their adjacent behavior as
the requested scientific contracts.

GitHub issue
[#571](https://github.com/edithatogo/voiage/issues/571) is the native
sub-issue of frontier programme issue
[#318](https://github.com/edithatogo/voiage/issues/318), under programme
[#313](https://github.com/edithatogo/voiage/issues/313), and is in
[Project 28](https://github.com/users/edithatogo/projects/28).

## Requirements

### R1 — Common study-design contract

- Represent feasible sample sizes or durations, allocation, study model,
  population/time horizon, uptake and delay, study and opportunity costs,
  constraints, guardrails and reproducibility settings.
- Require EVPI, EVSI and costs to use compatible units, population scaling,
  discounting and decision scope.
- Preserve signed ENBS and do not silently clamp economically unattractive
  designs to zero.

### R2 — COSS

- Evaluate EVSI, research cost and signed ENBS over a declared finite or
  discretized feasible design set.
- Return the full curve, feasible/infeasible flags, selected optimum, maximum
  ENBS, boundary-optimum diagnostics and a deterministic tie policy.
- Support non-monotone estimated curves and report estimator uncertainty around
  both ENBS and the selected design.
- Provide an accessible visualization while keeping the result contract
  independent from Matplotlib.

### R3 — EVSI/EVPI efficiency

- Return the dimensionless ratio \(EVSI/EVPI\) only when numerator and
  denominator share a decision problem, units and scaling.
- Define zero-EVPI behavior explicitly and reject incompatible or non-finite
  inputs.
- Report tolerance-aware diagnostics when Monte Carlo estimates fall slightly
  outside theoretical bounds; do not silently relabel or clamp materially
  invalid results.
- Keep the ratio distinct from `total_voi / total_cost`, return on investment,
  ENBS and cost-effectiveness ratios.

### R4 — Experiment portfolios and reconciliation

- Retain #571's portfolio requirements for shared traffic, duration, capacity,
  guardrails, interference, multiplicity, delayed effects and dependencies.
- Return portfolio allocation, stopping rules, gross/net EVSI and ENBS, policy
  changes and diagnostics.
- Audit `voiage.plot.plot_evsi_vs_sample_size` and
  `VOIBasedSampleSizeOptimizer`; reuse validated behavior but deprecate or
  rename misleading `voi_efficiency` semantics before exposing the new ratio.
- Keep experiment-platform adapters optional and outside the stable core.

### R5 — Assurance and language surfaces

- Use analytical or enumerable sample-size references and independent argmax
  checks.
- Cover interior, boundary, tied, infeasible and non-monotone optima; zero-EVPI
  and ratio-bound cases; unit/scaling mismatches; and serialization.
- Record estimator uncertainty, solver/optimizer diagnostics, provenance and
  performance evidence.
- Maintain explicit Rust/Python/R/Julia/Mojo capability dispositions.

## Acceptance criteria

- **AC-01:** A versioned design and cost contract makes EVPI, EVSI, ENBS and
  feasibility inputs commensurate.
- **AC-02:** COSS returns evaluated curves, deterministic optimum selection,
  tie/boundary diagnostics and uncertainty rather than only drawing a plot.
- **AC-03:** EVSI/EVPI has explicit zero-denominator, units, scaling,
  tolerance and bounds behavior and cannot be confused with value/cost.
- **AC-04:** Existing plotting and clinical-optimizer behavior is classified,
  tested and reconciled without an unsupported maturity promotion.
- **AC-05:** Analytical/enumerable, property, edge, error and optimizer tests
  cover the declared scientific envelope.
- **AC-06:** Runtime, schemas, diagnostics, CLI, reporting, plotting,
  provenance, examples and method registry agree.
- **AC-07:** Rust/Python/R/Julia/Mojo dispositions and shared fixtures are
  explicit.
- **AC-08:** GitHub #571, parent #318, programme #313, Project 28, this track
  and the central cross-reference manifest remain bidirectionally linked.
- **AC-09:** Automated review, full local validation, the repository harness
  and hosted required checks pass before repository completion.

## Non-functional constraints

- Stable numerical policy belongs in Rust; Python owns orchestration,
  reporting and plotting rather than duplicate kernels.
- Optimization must be deterministic for identical inputs and declared seed.
- Result contracts must be versioned, finite-validated and serializable.
- Existing stable EVPI, EVSI and ENBS behavior must remain backward compatible.

## External and human gates

- Scientific review must approve cost, scaling, tie and zero-EVPI semantics
  before stable promotion.
- Hosted checks, merge, release dispatch, registry publication and external
  platform adapters remain separate gates.
- Planning and a graphical example do not establish runtime completion.

## Out of scope

- Treating independent power calculations as VOI optimization.
- Hidden extrapolation beyond the declared feasible design set.
- Reusing `total_voi / total_cost` under the EVSI/EVPI name.
- Making an experiment-platform SDK a stable-core dependency.

## Authoritative inputs

- User-approved feature description in the 2026-07-27 Codex task.
- GitHub issue
  [#571](https://github.com/edithatogo/voiage/issues/571), live revision
  updated 2026-07-27.
- Frontier parent
  [#318](https://github.com/edithatogo/voiage/issues/318) and programme
  [#313](https://github.com/edithatogo/voiage/issues/313), live revisions
  updated 2026-07-27.
- `voiage/plot/voi_curves.py`, `voiage/clinical_trials.py`,
  `voiage/methods/sample_information.py` and `specs/v1/stable-api.json` at
  repository baseline `ceefb515`.
- `conductor/product.md`, `conductor/product-guidelines.md`,
  `conductor/tech-stack.md` and `conductor/workflow.md`.
