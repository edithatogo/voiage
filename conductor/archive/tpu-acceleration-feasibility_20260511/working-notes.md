# Working Notes: TPU Acceleration Feasibility

## Feasibility Gate Conditions

TPU production acceleration is still gated by staged evidence:

- shared acceleration abstraction contract is defined and requires contract-safe dense
  workloads with deterministic results,
- discrete/Apple path has no confirmed throughput gain in this branch,
- Colab v5e evidence now confirms TPU visibility and EVPI parity for the
  compact validation workload,
- no production-sized workload profile yet justifies the TPU compile/runtime
  overhead.

## Workload Suitability Criteria

Given the VOI kernels in-repo:

- dense tensor-style reductions are feasible candidates,
- dynamic sampling/control-flow-heavy methods remain poor TPU fit,
- only very large and regular workloads would make TPU compile and transfer costs
  credible.

Current status remains **feasibility hold** for production acceleration. The
runtime-evidence gap is closed for the compact validation workload by:

- `conductor/archive/hpc-acceleration-abstraction-contract_20260511/handoff/colab_tpu_accelerator_evidence.json`

## Contract Rule

Any TPU follow-on remains gated by CPU contract parity and explicit comparison
packets with identical schemas. The Colab v5e packet records
`jax_devices == ["TPU_0(process=0,(0,0,0,0))"]`, `jax_platforms == ["tpu"]`,
and `cpu_evpi == jax_evpi == 1.25`.

## Transition Decision

Current decision: **Do not promote TPU beyond the existing implementation track
boundary into production acceleration yet**. Re-evaluate only after a confirmed
discrete/Apple stage success or TPU-specific large workload packet with
reproducible speedup and workload regularity evidence.

## Decision Packet (Reviewer-ready)

Decision: `feasibility_hold`

- Source: `handoff/feasibility_decision.json`
- Dependency gate: approved `hpc_acceleration_abstraction_contract_20260511`
  (contract-preserve + CPU-authoritative rules), Colab v5e parity evidence
  captured, and upstream `discrete-gpu-acceleration_20260511` speedup gate
  still unresolved.
- Required to reopen: confirmed contract-safe, repeatable gain on dense
  production-sized workloads from a completed upstream hardware stage or a
  TPU-specific large workload packet.
