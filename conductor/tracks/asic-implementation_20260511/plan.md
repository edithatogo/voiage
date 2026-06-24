# Track Implementation Plan: ASIC Implementation

## Phase 1: Deployment Assumptions [checkpoint: ]

- [x] Task: Define the ASIC/custom-circuit deployment assumptions.
- [x] Task: Identify the workloads that justify ASIC exploration.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Deployment Assumptions' (Protocol in workflow.md)

## Phase 2: Implementation And Evidence [checkpoint: ]

- [x] Task: Expose the ASIC execution lane as an explicit adapter placeholder.
- [x] Task: Document the placeholder behavior in the CLI and developer guide.
- [x] Task: Implement the ASIC execution lane as a free pre-silicon evidence path.
    - [x] Reuse the deterministic fixed-point EVPI-style RTL kernel and CPU fixture.
    - [x] Add a Verilator testbench comparing the hardware kernel to CPU fixture values.
    - [x] Add a manifest generator that records input hashes, tool versions, command statuses, and output artifact paths.
- [x] Task: Add GitHub Actions and fallback-runner evidence options.
    - [x] Use GitHub Actions with Docker as the default free runner.
    - [x] Cover OpenROAD, OpenLane, SKY130, and RTL-to-GDS command plans in the manifest.
    - [x] Document GitHub Codespaces and Google Cloud Shell as manual fallback runners.
- [x] Task: Add CPU/ASIC pre-silicon comparison evidence and tests.
    - [x] Commit a probe manifest under this track's handoff directory.
    - [x] Add regression tests for the plan text, RTL assets, and manifest schema.
    - [x] Keep Tiny Tapeout, SkyWater MPW, and fabricated-silicon runtime as future external gates.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Implementation And Evidence' (Protocol in workflow.md)
