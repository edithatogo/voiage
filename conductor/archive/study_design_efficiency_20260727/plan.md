# Implementation Plan

## Phase 1 — Contract and legacy reconciliation [checkpoint: c8285d5]

- [x] **S1:** Freeze common study-design, cost, population, time-horizon,
  feasibility and reproducibility semantics. (AC-01) — `1037f06`
- [x] **S2:** Freeze the exact COSS result contract: evaluated design records,
  feasible range/set, signed ENBS curve, optimum, deterministic tie policy,
  boundary state, uncertainty around the optimum and plotting inputs. (AC-02)
  — `e930186`
- [x] **S3:** Freeze EVSI/EVPI units, scaling, zero-denominator, tolerance and
  bounds behavior. (AC-03) — `97ec51f`
- [x] **S4:** Audit the existing plot and clinical optimizer, including the
  misnamed value/cost `voi_efficiency` field, and record compatibility or
  deprecation decisions. (AC-04) — `a747fb0`
- [x] **S5:** Run automated contract review and full Conductor validation.
  (AC-01–AC-04, AC-08) — `9314fb6`

## Phase 2 — Reference evidence before implementation [checkpoint: c357ebe]

- [x] **S6:** Add failing analytical/enumerable COSS and signed-ENBS reference
  tests with independent argmax calculations. (AC-02, AC-05) — `a14d3a2`
- [x] **S7:** Add failing tests for interior, boundary, tied, infeasible and
  non-monotone design curves. (AC-02, AC-05) — `e016589`
- [x] **S8:** Add failing EVSI/EVPI tests for common scaling, zero EVPI,
  theoretical limits, Monte Carlo tolerance and invalid inputs. (AC-03,
  AC-05) — `4b24b88`
- [x] **S9:** Review reference independence, cost provenance and estimator
  uncertainty requirements. (AC-05) — `c7a0399`

## Phase 3 — Runtime and optimizer [checkpoint: 136b9b2]

- [x] **S10:** Implement versioned Rust result envelopes and accepted numerical
  kernels for COSS and EVSI/EVPI, or record a reviewed exclusion. (AC-02,
  AC-03, AC-06) — `bf908ed`
- [x] **S11:** Implement the thin Python orchestration façade and deterministic
  design optimizer without numerical fallback. (AC-01, AC-02, AC-06) —
  `9f22894`
- [x] **S12:** Reconcile or deprecate legacy optimizer behavior and preserve
  stable API compatibility. (AC-04) — `a68157c`
- [x] **S13:** Add property, differential, serialization, uncertainty,
  optimizer and benchmark assurance. (AC-05, AC-09) — `3548c37`
- [x] **S14:** Run automated implementation review, focused tests and the
  repository harness. (AC-05, AC-09) — `c60a2ee`

## Phase 4 — Portfolio, user and binding surfaces [checkpoint: b11377c]

- [x] **S15:** Integrate single-study COSS and efficiency outputs with #571's
  experiment-portfolio allocation and constraint contract. (AC-01, AC-06) —
  `3030636`
- [x] **S16:** Add CLI, reporting, provenance and accessible plotting/examples
  that consume the versioned result contract. (AC-06) — `9e74de2`
- [x] **S17:** Record Rust, Python, R, Julia and Mojo capability dispositions
  and shared-fixture evidence or explicit unsupported failures. (AC-07) —
  `41dc45f`
- [x] **S18:** Reconcile roadmap, registry, documentation, v1.2.0 MoSCoW
  requirements, Mermaid design, canonical C16, GitHub, Project 28, Conductor
  and hosted evidence. (AC-06–AC-10) — `7fadc1c`
- [x] **S19:** Run final automated review, full local validation and hosted
  required checks. PR #679 final exact head
  `ce5d712779897bdd7d398e367de6a7e0bc743692` completed 65 terminal
  conclusions (60 successes, four governed skips and one neutral conclusion),
  with no pending or failed checks; both review threads were resolved before
  squash merge `5d059a80447afc85cee63eb85971fc1c9e80f40c` at
  `2026-07-31T15:38:47Z`. Only delivery subissues #680, #681 and #682 are
  closure-eligible. Scientific review, Rust/R/Julia parity, stable promotion,
  release, parent #571 closure and umbrella #318 closure remain pending.
  (AC-09) — `a9e626f`

## Phase 5 — Orchestrated scientific remediation and re-review

- [x] **S20:** **Legacy follow-up (not part of completed track acceptance):** Freeze the candidate and run the umbrella Phase 5 four-role
  panel against COSS, ENBS and EVSI/EVPI efficiency. (M15-S7–M15-S11; AC-01–
  AC-05, AC-09)
- [x] **S21:** Separate complete-enumeration and evaluated-set results; add the
  no-study comparator, economic viability, sampling recommendation, boundary
  sensitivity and regret semantics without changing the curve argmax.
  (M15-S7, M15-S8, M15-S11; AC-02, AC-05, AC-06) `b3fb0b8`
- [x] **S22:** Bind selection uncertainty to replayable joint replicates and
  add paired efficiency uncertainty, calibration, near-tie and winner's-curse
  evidence. (M15-S9–M15-S11; AC-02, AC-03, AC-05) `d926e31`
- [~] **S23:** Add portable request/result schemas and capability metadata,
  correct the `1.0`/`1.0.0` contract-version mismatch, execute installed-wheel
  fixtures and downgrade any unsupported `fixture-backed` claim until the
  evidence exists. (AC-02, AC-06–AC-10)
- [x] **S24:** **Legacy follow-up (not part of completed track acceptance):** Remediate through nested #571 issues, rebind the candidate and
  obtain affected-role plus named independent human re-review; retain parity,
  promotion, release and closure as separate gates. (AC-01, AC-05, AC-09)

<!-- Historical contract markers retained for panel-corpus tests: - [ ] **S20:** - [ ] **S21:** - [ ] **S22:** - [ ] **S23:** - [ ] **S24:** -->
