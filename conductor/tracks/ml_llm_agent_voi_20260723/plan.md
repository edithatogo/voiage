# Track Implementation Plan: ML, LLM, And Agent Value Of Information

## Phase 1: Decision and backend contracts

- [ ] Add failing utility, EIG-versus-VOI, protocol, privacy, and fallback tests.
- [ ] Define prediction, posterior-update, utility, acquisition, provenance, and
  stopping contracts.
- [ ] Freeze `policy-uplift-voi` under
  [#578](https://github.com/edithatogo/voiage/issues/578), including treatment
  eligibility, heterogeneous and delayed effects, action cost, capacity,
  fairness, interference, fatigue, implementation delay, causal-model
  uncertainty, off-policy support, information action, policy change, and
  gross/net decision value.
- [ ] Freeze `decision-focused-model-value` under
  [#576](https://github.com/edithatogo/voiage/issues/576), including competing
  model, human and workflow policies; downstream utility and regret; model and
  data costs; latency; constraints; drift; calibration; monitoring; and
  separation of model value from information value.
- [ ] Threat-model prompt injection, data/retrieval poisoning, tool
  exfiltration, reward hacking, evaluation contamination, adaptive overfitting,
  correlated judge/verifier failures, provider drift, and human override.
- [ ] Register ML/LLM/agent methods at experimental maturity.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 1: Decision and backend contracts'
  (Protocol in workflow.md).

## Phase 2: Methods and contexts

- [ ] Implement deterministic core estimators and optional backend adapters.
- [ ] Implement label, model, retrieval, compute, escalation, drift, tool, and
  delegation decision contexts.
- [ ] Implement #578 with deterministic finite and simulation references,
  policy and constraint switches, treatment-effect and support diagnostics,
  estimator assurance, and a churn/retention adapter that reuses the common
  decision engine.
- [ ] Implement #576 with policy-aware model comparison, acquisition and
  refresh value, regret decomposition, calibration/shift sensitivity, and
  explicit human/model combination policies.
- [ ] Add calibration, sensitivity, stopping, privacy, and provenance results.
- [ ] Add adversarial-information, safety-constraint, dependent-evaluator, and
  escalation/abstention analyses when they change net information value.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Automated review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 2: Methods and contexts'
  (Protocol in workflow.md).

## Phase 3: Evidence and maturity

- [ ] Execute offline synthetic and rights-cleared public examples.
- [ ] Verify #576 and #578 through the public registry, portable schema,
  advertised language surfaces, adversarial fixtures, and independent
  references before any maturity promotion.
- [ ] Run determinism, backend, Rust, binding, docs, security, and full gates.
- [ ] Reconcile experimental status and promotion requirements.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - User Manual Verification 'Phase 3: Evidence and maturity'
  (Protocol in workflow.md).
