# Track Specification: ASIC Implementation

## Overview

This track turns ASIC/custom-circuit work into an implementation lane rather
than a permanent feasibility hold. It should still be benchmark-gated and
contract-preserving.

## Functional Requirements

1. Define the custom-circuit deployment assumptions.
2. Preserve the CPU contract and public API.
3. Keep the path optional and justified by evidence.
4. Treat free CI-based pre-silicon evidence as acceptable first-pass progress:
   Verilator simulation/lint, Yosys synthesis, and OpenROAD/OpenLane/SKY130
   RTL-to-GDS planning.
5. Keep Tiny Tapeout, SkyWater MPW, and fabricated-silicon runtime as separate
   future external gates.

## Acceptance Criteria

1. ASIC/custom-circuit pre-silicon evidence is possible behind the shared
   abstraction without changing the public API.
2. CPU/ASIC comparison evidence is reproducible from committed RTL, fixtures,
   and manifests.
3. The public contract remains stable.
4. Any fabricated-silicon claim is explicitly deferred until shuttle or silicon
   access exists.
