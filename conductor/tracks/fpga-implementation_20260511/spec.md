# Track Specification: FPGA Implementation

## Overview

This track captures FPGA-style acceleration as an explicit implementation lane
instead of folding it into ASIC/custom-circuit feasibility language.

## Functional Requirements

1. Define the FPGA toolchain and kernel shape.
2. Preserve the CPU contract and output envelope.
3. Keep the path optional and benchmark-driven.
4. Treat free CI-based pre-silicon evidence as acceptable first-pass progress:
   Verilator simulation/lint, Yosys synthesis, and nextpnr place-and-route.
5. Keep physical FPGA board runtime as a separate future evidence gate.

## Acceptance Criteria

1. FPGA pre-silicon evidence is available behind the shared abstraction without
   changing the public API.
2. CPU/FPGA comparison packets are reproducible from committed RTL, fixtures,
   and manifests.
3. The public API remains stable.
4. Any physical-board claim is explicitly deferred until hardware access exists.
