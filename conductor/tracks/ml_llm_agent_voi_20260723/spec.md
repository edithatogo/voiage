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

## Architecture, privacy, and compatibility

Use offline tables and CPU deterministic references. PyTorch, JAX, Pyro,
BoTorch, Hugging Face, and provider SDKs are named extras only. No network or
private-data transmission is required. Backends use versioned protocols and
fail explicitly.

## Acceptance criteria

Every method and example satisfies the formal decision contract, calibration,
cost/latency/privacy sensitivity, drift, stopping, determinism, and fallback
tests. Entropy-only scores are never labelled economic VOI.

