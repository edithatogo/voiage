# Track Specification: ML, LLM, And Agent Value Of Information

## Overview

Add backend-neutral decision-VOI and information-design methods for machine
learning, foundation models, LLMs, and agents.

## Requirements

1. Distinguish downstream decision VOI from entropy/mutual-information EIG.
2. Cover BOED, EIG/EPIG, active/batch/cost-aware learning, knowledge gradient,
   multi-fidelity/task VOI, data/feature/test acquisition, model evaluation,
   simulation-based inference, amortized EVSI, value of computation, drift and
   refresh, fairness/privacy/safety information, escalation, and sequential
   learning decisions.
3. Cover model/provider routing; prompts/tools/workflows; self-consistency,
   critique, verification and judging; test-time stopping; RAG; human feedback;
   tuning/distillation/quantization; eval/red-team acquisition; agent tool calls
   and delegation; hallucination/citation/safety review; and monitoring.
4. Require explicit alternatives, uncertainty, information action, predictive
   model, utility/loss, current and posterior decisions, cost, net value,
   diagnostics, and stopping rule.
5. Model prompt injection, poisoning, exfiltration, reward hacking, evaluation
   contamination, adaptive overfitting, correlated evaluator failures,
   provider drift, and human override whenever they affect action utility.

## Industry decision-value workstreams

GitHub [#578](https://github.com/edithatogo/voiage/issues/578), record ID
`policy-uplift-voi`, must value information through the treatment, offer,
retention, pricing, collection, approval, allocation, or escalation policy it
changes. The contract includes eligibility, heterogeneous treatment effects,
action costs, capacity, contact fatigue, fairness, interference, delayed
outcomes, off-policy support, implementation delay, and uncertainty in both
outcome and causal-effect models. Predictive lift, uplift accuracy, and policy
value without an information action are not VOI.

GitHub [#576](https://github.com/edithatogo/voiage/issues/576), record ID
`decision-focused-model-value`, must compare models, forecasts, rules, prompts,
agents, and human/model combinations by downstream expected utility, regret,
constraints, latency, monitoring burden, acquisition/serving cost, and the
policy changes caused by information. It must support calibration and
distribution-shift sensitivity and must keep model selection, model-risk
assurance, information acquisition, and implementation value distinct.

Both workstreams reuse the canonical Decision Problem and estimator-assurance
envelopes. Customer churn is a first application, not a separate numerical
engine: the information action can be a new label, survey, contact outcome,
feature, experiment, model refresh, or analyst review, and value is measured
through the feasible retention policy.

## Architecture, privacy, and compatibility

Use offline tables and CPU deterministic references. PyTorch, JAX, Pyro,
BoTorch, Hugging Face, and provider SDKs are named extras only. No network or
private-data transmission is required. Backends use versioned protocols and
fail explicitly.

## Acceptance criteria

Every method and example satisfies the formal decision contract, calibration,
cost/latency/privacy sensitivity, drift, stopping, determinism, and fallback
tests. Issues #576 and #578 have executable decision-focused or policy-uplift
evidence, or an explicit reviewed disposition. Entropy-only, predictive-score,
or uplift-accuracy metrics are never labelled economic VOI.
