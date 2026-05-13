# Working Notes: TPU Acceleration Feasibility

## Feasibility Gate Conditions

TPU path is blocked by the current staged evidence:

- shared acceleration abstraction contract is defined and requires contract-safe dense
  workloads with deterministic results,
- discrete/Apple path has no confirmed throughput gain in this branch,
- no workload profile yet justifies the TPU compile/runtime overhead.

## Workload Suitability Criteria

Given the VOI kernels in-repo:

- dense tensor-style reductions are feasible candidates,
- dynamic sampling/control-flow-heavy methods remain poor TPU fit,
- only very large and regular workloads would make TPU compile and transfer costs
  credible.

Current status remains **feasibility hold**.

## Contract Rule

Any TPU follow-on remains gated by CPU contract parity and explicit comparison packets
with identical schemas.

## Transition Decision

Current decision: **Do not open a TPU implementation track yet**. Re-evaluate only
after a confirmed discrete/Apple stage success with reproducible speedup and workload
regularity evidence.

## Decision Packet (Reviewer-ready)

Decision: `feasibility_hold`

- Source: `handoff/feasibility_decision.json`
- Dependency gate: approved `hpc_acceleration_abstraction_contract_20260511`
  (contract-preserve + CPU-authoritative rules) and upstream `discrete-gpu-acceleration_20260511`
  evidence gate still unresolved.
- Required to reopen: confirmed contract-safe, repeatable gain on dense workloads
  from a completed upstream hardware stage.
