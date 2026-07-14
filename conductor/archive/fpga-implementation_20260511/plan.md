# Track Implementation Plan: FPGA Implementation

## Phase 1: Toolchain And Kernel Definition [checkpoint: ]

- [x] Task: Define the FPGA toolchain and deployment assumptions.
- [x] Task: Identify the workloads that could be ported to FPGA.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Toolchain And Kernel Definition' (Protocol in workflow.md)

## Phase 2: Implementation And Validation [checkpoint: ]

- [x] Task: Expose the FPGA execution lane as an explicit adapter placeholder.
- [x] Task: Document the placeholder behavior in the CLI and developer guide.
- [x] Task: Implement the FPGA execution lane as a free pre-silicon evidence path.
    - [x] Add a deterministic fixed-point EVPI-style RTL kernel and CPU fixture.
    - [x] Add a Verilator testbench comparing the hardware kernel to CPU fixture values.
    - [x] Add a manifest generator that records input hashes, tool versions, command statuses, and output artifact paths.
- [x] Task: Add GitHub Actions and fallback-runner evidence options.
    - [x] Use GitHub Actions with OSS CAD Suite as the default free runner.
    - [x] Cover Verilator, Yosys, and nextpnr command plans in the manifest.
    - [x] Document GitHub Codespaces and Google Cloud Shell as manual fallback runners.
- [x] Task: Add CPU/FPGA pre-silicon comparison evidence and tests.
    - [x] Commit a probe manifest under this track's handoff directory.
    - [x] Add regression tests for the plan text, RTL assets, and manifest schema.
    - [x] Keep physical FPGA board runtime as a future external gate.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Implementation And Validation' (Protocol in workflow.md)
