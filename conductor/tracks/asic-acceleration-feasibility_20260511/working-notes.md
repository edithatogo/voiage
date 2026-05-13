# Working Notes: ASIC / Custom-Circuit Feasibility

## Evidence and Positioning

ASIC adoption requires sustained regular dense kernels and a large volume of evidence.
Current stage remains **analysis-only** because:

- GPU/TPU feasibility preconditions are not yet satisfied,
- no confirmed large-throughput proof exists to justify a hardening investment,
- project policy keeps CPU and shared abstraction contracts as the reference path.

## Workload Shape Requirements

The only class that could justify ASIC follow-on work is highly regular dense
computation with predictable tensor sizes and minimal host-device interaction.

## Decision

Current decision: **stay at feasibility/feasibility-review only**.
No implementation track will be opened until both workload regularity and lower-level
benchmarks from earlier tracks show sustained, validated gains.

## Decision Packet (Reviewer-ready)

Decision: `feasibility_hold`

- Source: `handoff/feasibility_decision.json`
- Dependency gate: approved `hpc_acceleration_abstraction_contract_20260511`
  plus the completed `tpu-acceleration-feasibility_20260511` and
  `discrete-gpu-acceleration_20260511` evidence gates.
- Required to reopen: sustained benchmark evidence that passes CPU-contract-preserving
  comparison packets and demonstrates ASIC/ASIC-ready regularity economics.

## Contract Rule

No API changes are permitted; ASIC is only valid if CPU contract and result payload
shape remain authoritative and preserved.
