# Working Notes: HPC Capability Implementation Program

## Implementation Order

1. CPU cluster and local parallelism
2. Apple Metal hardening
3. Discrete GPU implementation
4. TPU implementation
5. FPGA implementation
6. ASIC implementation

## Constraint

All child tracks must preserve the existing CPU-first contract and the current
public API shape.
