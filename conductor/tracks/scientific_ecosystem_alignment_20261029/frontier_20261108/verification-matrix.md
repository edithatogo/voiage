# Verification matrix and benchmark admission

Status: candidate case families, not acquired datasets or passed evaluations.
Source links and their limitations are recorded in [sources](sources.md).
Q01 must reconcile this matrix with every advertised method/estimator/feature;
an unmatched capability remains unverified or explicitly excluded.

## Complete decision cases

Every admitted case records: case/version; estimand and assumptions; actions and
utility/loss; joint uncertainty, weights and units; study likelihood and posterior
contract where applicable; time/population/perspective; independent reference and
its error budget; source and generator revisions; checksums; rights/redistribution;
seed and execution environment; transformations; intended failure cases; and
permitted claims. Raw observations alone cannot verify VOI.

| ID | Methods/features | Candidate case or data | Independent witness and limits |
| --- | --- | --- | --- |
| D01 | EVPI, ENBS, CEAC/CEAF, dominance | Exact finite payoff tables; BCEA Smoking/Vaccine examples (S02) | Enumeration for exact cases; separate equations/package comparison for realistic cases. Distinguish PSA example data from patient observations. |
| D02 | Linear/nonlinear EVPPI | `voi` chemotherapy inputs/model functions (S01); synthetic interactions and correlated inputs | Nested reference or controlled quadrature; held-out fit and decision error. No unconditional independence assumptions. |
| D03 | Analytical/nested EVSI | Normal–normal, beta–binomial, gamma–Poisson synthetic studies; chemotherapy (S01) | Closed form, exact finite sum or bounded quadrature/truncation; coherent joint updating. |
| D04 | Regression/moment/importance EVSI | Identical D02/D03 study problems across estimators | Pinned `voi` implementations plus independent reference; expose weighting failures, bias and Monte Carlo error. |
| D05 | Survival and follow-up EVSI | Generated known-truth censored survival studies; a published reproduction selected during source admission | Verify censoring, extrapolation, waning and follow-up decisions. Literature/code/rights selection is still pending; no named dataset claimed acquired. |
| D06 | Calibration and structural VOI | DARTH Sick–Sicker and calibration workflow (S03), plus specified alternative model structures | Preserve calibrated joint uncertainty; independent calibration targets and enumerated small structure mixtures. |
| D07 | Network-meta-analysis VOI | `gemtc` smoking network candidate (S04) plus a declared economic model | Preserve multi-arm dependence and network structure. Treatment-effect data alone do not define the research decision. |
| D08 | Sequential/adaptive/flexibility VOI | Finite-horizon beta–binomial decision processes | Independent enumeration/dynamic programming of stopping and actions; exact only for declared finite cases. |
| D09 | Design portfolios/budget/capacity | Small enumerated research portfolios with shared evidence and finite constraints | Exact optimum and regret; explicit no-study option. Never add correlated study values as if independent. |
| D10 | Equity, heterogeneity, perspective, MCDA | Synthetic subgroup/perspective tables; existing VOP application | Independent aggregation and ranking; normative weights are declared assumptions, not empirically verified truth. |
| D11 | Utility information/pricing/signed social value | Finite affine, exponential/log utility cases and explicit strategic examples | Check domain, utility normalisation, roots, units and stated information-order assumptions. Negative social/strategic value is not automatically an error. |
| D12 | Adoption/implementation VOI | Existing Innovate Bass pilot; held-out OWID uptake series (S06) after rights checks | First verify forecasts, then a complete joint decision model. Observed uptake is not causal evidence of implementation effectiveness. |
| D13 | Observational/causal/transportability VOI | Known-mechanism synthetic causal models with controlled bias and transfer | Structural ground truth and sensitivity to misspecification; public observational data cannot identify all assumptions. |
| D14 | Prediction-model and policy-learning VOI | Generated prediction/validation problems with known mechanisms; published cases selected later | Held-out clinical/policy decision utility and study choice, not discrimination alone. Source reproduction remains pending. |
| D15 | Estimation-focused VOI | Conjugate posterior-variance examples and `voi` estimation functions (S01) | Exact expected variance reduction; distinguish variance reduction from payoff-based VOI. |
| D16 | Amortised/SBI/multifidelity/decision-focused computation | `sbibm` reference tasks (S05) augmented with explicit decisions; paired cheap/accurate simulators | Check reference-posterior quality, approximation-induced decision regret and amortisation break-even; held-out tasks, no tuning on confirmatory references. |
| D17 | Ingestion/provenance/bindings | Existing Arrow/Parquet, Croissant/Frictionless, C ABI and installed-consumer fixtures | Lost metadata, invalid weights/units, changed schema, buffer lifetime and reordered labels; each language must report unsupported methods honestly. |
| D18 | Parallelism/restart/CPU/Metal | D01–D04 at prespecified sizes and precisions plus controlled interruption fixtures | Determinism or declared tolerance, memory/time accounting, restart equivalence, cold/warm and transfer overhead, explicit CPU fallback. |

## Admission and licensing

All external entries begin as **candidate-source-only**, with revision, checksum and
redistribution disposition unset until acquisition. Documentation access is not a
licence grant or proof of reproducibility. Check code, data and derived-artifact
rights separately; where redistribution is restricted, retain lawful acquisition
instructions and checksums rather than copying data. Synthetic fixtures also need
an owned generator, seed and declared licence. No patient-level uploads are planned.

Reuse the existing registry and manifest schemas where sufficient; amend them
through their owner if required. Do not replace a fixture-only registry label with
'validated' simply because a public source was found. Offline CI consumes admitted,
versioned artifacts rather than downloading mutable upstream data on every run.

## Prespecified evaluation

Report bias/RMSE, uncertainty interval coverage where supported, probability of a
wrong intervention/study, selected-decision regret, convergence/diagnostic failures,
wall time and peak memory. Distinguish numerical error, estimator approximation,
Monte Carlo error and model misspecification. Give confidence intervals and all
attempted cases, including failures/timeouts. References require their own error
bounds; another package is a comparator, not unquestionable truth.

Freeze case-specific tolerances and scientifically meaningful decision margins
before confirmatory runs. Use existing exact-case numerical tolerances where
applicable; do not introduce a single arbitrary percentage threshold across units.
Use paired randomness where it improves comparison, disclose dependence, and
choose replication counts from the desired Monte Carlo precision. Keep development
and confirmation cases separate. Investigate near-zero denominators rather than
reporting misleading percentage errors.

Keep three comparisons separate: identical estimator/different implementation;
different estimator/matched accuracy; identical workflow/different hardware. Record
input sizes, CPU/device, compiler/runtime, thread count, precision, warm-up, transfer
and compilation costs, failures and package revisions. Energy is optional and only
reported with a defensible measurement method. Benchmark budgets are not speed measurements.

## Storage and reproduction

GitHub owns protocols, generators, small exact fixtures and adapters. A proposed
`edithatogo/voiage-benchmarks` Hugging Face dataset can hold larger rights-cleared
Parquet cases/results after namespace, existing-resource and licence checks; its
existence or creation is not claimed. Dataset cards must state decisions, assumptions,
limitations, synthetic status and splits. A release bundle should pin code, case
manifest, environment, outputs and report to an immutable DOI-backed archive.
No Hub or archive upload occurs in this planning change.
